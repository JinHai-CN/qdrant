use std::collections::HashMap;

use itertools::Itertools;

use crate::id_tracker::IdTrackerSS;
use crate::index::field_index::FieldIndex;
use crate::index::query_optimization::nested_filter::NestedConditionCheckerResult::{
    Matches, NoMatch,
};
use crate::index::query_optimization::optimized_filter::ConditionCheckerFn;
use crate::index::query_optimization::optimizer::IndexesMap;
use crate::index::query_optimization::payload_provider::PayloadProvider;
use crate::payload_storage::nested_query_checker::{
    check_nested_is_empty_condition, check_nested_is_null_condition, nested_check_field_condition,
};
use crate::types::{
    AnyVariants, Condition, FieldCondition, FloatPayloadType, GeoBoundingBox, GeoRadius, Match,
    MatchAny, MatchText, MatchValue, PointOffsetType, Range, ValueVariants,
};

/// Given a point_id, returns the list of nested paths matching the condition
pub type NestedConditionCheckerFn<'a> =
    Box<dyn Fn(PointOffsetType) -> NestedConditionCheckerResult + 'a>;

pub type NestedPath = String;

pub enum NestedConditionCheckerResult {
    NoMatch,
    Matches(Vec<NestedPath>),
}

impl NestedConditionCheckerResult {
    pub fn from_matches(matches: Vec<String>) -> Self {
        if matches.is_empty() {
            NoMatch
        } else {
            Matches(matches)
        }
    }

    pub fn from_indexes(indexes: impl Iterator<Item = usize>) -> Self {
        let values: Vec<String> = indexes.map(|i| i.to_string()).collect();
        if values.is_empty() {
            NoMatch
        } else {
            Matches(values)
        }
    }
}

/// Merge several nested condition results into a single regular condition checker
///
/// return a single condition checker that will return true if all nested condition checkers for the point_id
pub fn merge_nested_condition_checkers(
    nested_checkers: Vec<NestedConditionCheckerFn>,
) -> ConditionCheckerFn {
    Box::new(move |point_id: PointOffsetType| {
        // number of nested condition checkers
        let condition_count = nested_checkers.len();
        // binds path to the match count
        let mut matches: HashMap<NestedPath, usize> = HashMap::new();
        for nested_checker in &nested_checkers {
            let result = nested_checker(point_id);
            match result {
                NoMatch => (),
                Matches(mut nested_match_paths) => {
                    // increment match count for each nested match
                    for nested_match in nested_match_paths.drain(..) {
                        let count = matches.entry(nested_match).or_insert(0);
                        *count += 1;
                    }
                }
            }
        }
        // if any of the nested path is matching for each nested condition
        // then the point_id matches and matching synthetic `ConditionCheckerFn can be created`
        matches.iter().any(|(_, count)| *count == condition_count)
    })
}

pub fn nested_condition_converter<'a>(
    condition: &'a Condition,
    field_indexes: &'a IndexesMap,
    payload_provider: PayloadProvider,
    _id_tracker: &IdTrackerSS,
    nested_path: &'a str,
) -> NestedConditionCheckerFn<'a> {
    match condition {
        Condition::Field(field_condition) => {
            let full_path = format!("{}.{}", nested_path, field_condition.key);
            field_indexes
                .get(&full_path)
                .and_then(|indexes| {
                    indexes
                        .iter()
                        .filter_map(|index| nested_field_condition_index(index, field_condition))
                        .next()
                })
                .unwrap_or_else(|| {
                    Box::new(move |point_id| {
                        payload_provider.with_payload(point_id, |payload| {
                            let matches = nested_check_field_condition(
                                field_condition,
                                &payload,
                                nested_path,
                            );
                            NestedConditionCheckerResult::from_matches(matches)
                        })
                    })
                })
        }
        Condition::IsEmpty(is_empty) => Box::new(move |point_id| {
            payload_provider.with_payload(point_id, |payload| {
                let matches = check_nested_is_empty_condition(nested_path, is_empty, &payload);
                NestedConditionCheckerResult::from_matches(matches)
            })
        }),
        Condition::IsNull(is_null) => Box::new(move |point_id| {
            payload_provider.with_payload(point_id, |payload| {
                let matches = check_nested_is_null_condition(nested_path, is_null, &payload);
                NestedConditionCheckerResult::from_matches(matches)
            })
        }),
        Condition::HasId(_) => unreachable!(), // TODO
        Condition::Nested(_) => unreachable!(),
        Condition::Filter(_) => unreachable!(),
    }
}

pub fn nested_field_condition_index<'a>(
    index: &'a FieldIndex,
    field_condition: &FieldCondition,
) -> Option<NestedConditionCheckerFn<'a>> {
    if let Some(checker) = field_condition
        .r#match
        .clone()
        .and_then(|cond| get_nested_match_checkers(index, cond))
    {
        return Some(checker);
    }

    if let Some(checker) = field_condition
        .range
        .clone()
        .and_then(|cond| get_nested_range_checkers(index, cond))
    {
        return Some(checker);
    }

    if let Some(checker) = field_condition
        .geo_radius
        .clone()
        .and_then(|cond| get_nested_geo_radius_checkers(index, cond))
    {
        return Some(checker);
    }

    if let Some(checker) = field_condition
        .geo_bounding_box
        .clone()
        .and_then(|cond| get_nested_geo_bounding_box_checkers(index, cond))
    {
        return Some(checker);
    }

    None
}

pub fn get_nested_geo_radius_checkers(
    index: &FieldIndex,
    geo_radius: GeoRadius,
) -> Option<NestedConditionCheckerFn> {
    match index {
        FieldIndex::GeoIndex(geo_index) => Some(Box::new(move |point_id: PointOffsetType| {
            match geo_index.get_values(point_id) {
                None => NoMatch,
                Some(values) => {
                    NestedConditionCheckerResult::from_indexes(values.iter().positions(
                        |geo_point| geo_radius.check_point(geo_point.lon, geo_point.lat),
                    ))
                }
            }
        })),
        _ => None,
    }
}

pub fn get_nested_geo_bounding_box_checkers(
    index: &FieldIndex,
    geo_bounding_box: GeoBoundingBox,
) -> Option<NestedConditionCheckerFn> {
    match index {
        FieldIndex::GeoIndex(geo_index) => Some(Box::new(move |point_id: PointOffsetType| {
            match geo_index.get_values(point_id) {
                None => NoMatch,
                Some(values) => {
                    NestedConditionCheckerResult::from_indexes(values.iter().positions(
                        |geo_point| geo_bounding_box.check_point(geo_point.lon, geo_point.lat),
                    ))
                }
            }
        })),
        _ => None,
    }
}

pub fn get_nested_range_checkers(
    index: &FieldIndex,
    range: Range,
) -> Option<NestedConditionCheckerFn> {
    match index {
        FieldIndex::IntIndex(num_index) => Some(Box::new(move |point_id: PointOffsetType| {
            match num_index.get_values(point_id) {
                None => NoMatch,
                Some(values) => NestedConditionCheckerResult::from_indexes(
                    values
                        .iter()
                        .copied()
                        .positions(|i| range.check_range(i as FloatPayloadType)),
                ),
            }
        })),
        FieldIndex::FloatIndex(num_index) => Some(Box::new(move |point_id: PointOffsetType| {
            match num_index.get_values(point_id) {
                None => NoMatch,
                Some(values) => NestedConditionCheckerResult::from_indexes(
                    values.iter().copied().positions(|i| range.check_range(i)),
                ),
            }
        })),
        _ => None,
    }
}

pub fn get_nested_match_checkers(
    index: &FieldIndex,
    cond_match: Match,
) -> Option<NestedConditionCheckerFn> {
    match cond_match {
        Match::Value(MatchValue {
            value: value_variant,
        }) => match (value_variant, index) {
            (ValueVariants::Keyword(keyword), FieldIndex::KeywordIndex(index)) => {
                Some(Box::new(move |point_id: PointOffsetType| {
                    match index.get_values(point_id) {
                        None => NoMatch,
                        Some(values) => NestedConditionCheckerResult::from_indexes(
                            values.iter().positions(|k| k == &keyword),
                        ),
                    }
                }))
            }
            (ValueVariants::Integer(value), FieldIndex::IntMapIndex(index)) => {
                Some(Box::new(move |point_id: PointOffsetType| {
                    match index.get_values(point_id) {
                        None => NoMatch,
                        Some(values) => NestedConditionCheckerResult::from_indexes(
                            values.iter().positions(|i| i == &value),
                        ),
                    }
                }))
            }
            _ => None,
        },
        Match::Text(MatchText { text }) => match index {
            FieldIndex::FullTextIndex(full_text_index) => {
                let parsed_query = full_text_index.parse_query(&text);
                Some(Box::new(
                    move |point_id: PointOffsetType| match full_text_index.get_doc(point_id) {
                        None => NoMatch,
                        Some(doc) => {
                            let _res = parsed_query.check_match(doc);
                            // TODO what to do with the result, there is no nesting here
                            NoMatch
                        }
                    },
                ))
            }
            _ => None,
        },
        Match::Any(MatchAny { any }) => match (any, index) {
            (AnyVariants::Keywords(list), FieldIndex::KeywordIndex(index)) => {
                Some(Box::new(move |point_id: PointOffsetType| {
                    match index.get_values(point_id) {
                        None => NoMatch,
                        Some(values) => NestedConditionCheckerResult::from_indexes(
                            values.iter().positions(|k| list.contains(k)),
                        ),
                    }
                }))
            }
            (AnyVariants::Integers(list), FieldIndex::IntMapIndex(index)) => {
                Some(Box::new(move |point_id: PointOffsetType| {
                    match index.get_values(point_id) {
                        None => NoMatch,
                        Some(values) => NestedConditionCheckerResult::from_indexes(
                            values.iter().positions(|i| list.contains(i)),
                        ),
                    }
                }))
            }
            _ => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_matching_merge_nested_condition_checkers() {
        let mut checkers: Vec<NestedConditionCheckerFn> = Vec::new();
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec![])
        }));
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec![])
        }));
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec![])
        }));

        let merged = merge_nested_condition_checkers(checkers);
        // none of the conditions are matching anything
        let result: bool = merged(0);
        assert!(!result);
    }

    #[test]
    fn single_matching_merge_nested_condition_checkers() {
        let mut checkers: Vec<NestedConditionCheckerFn> = Vec::new();
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec!["0".to_string()])
        }));
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec!["0".to_string()])
        }));
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec!["0".to_string()])
        }));

        let merged = merge_nested_condition_checkers(checkers);
        let result: bool = merged(0);
        assert!(result);
    }

    #[test]
    fn single_non_matching_merge_nested_condition_checkers() {
        let mut checkers: Vec<NestedConditionCheckerFn> = Vec::new();
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec!["0".to_string()])
        }));
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec!["0".to_string()])
        }));
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec!["1".to_string()])
        }));

        let merged = merge_nested_condition_checkers(checkers);
        // does not because all the checkers are not matching the same path
        let result: bool = merged(0);
        assert!(!result);
    }

    #[test]
    fn many_matching_merge_nested_condition_checkers() {
        let mut checkers: Vec<NestedConditionCheckerFn> = Vec::new();
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec!["0".to_string(), "1".to_string()])
        }));
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec!["0".to_string(), "1".to_string()])
        }));
        checkers.push(Box::new(|_point_id: PointOffsetType| {
            NestedConditionCheckerResult::from_matches(vec!["0".to_string()])
        }));

        let merged = merge_nested_condition_checkers(checkers);
        // still matching because of the path '1' matches all conditions
        let result: bool = merged(0);
        assert!(result);
    }
}
