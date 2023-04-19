use std::collections::HashMap;

use itertools::Itertools;

use crate::id_tracker::IdTrackerSS;
use crate::index::field_index::FieldIndex;
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

/// Payload element index
pub type ElemIndex = usize;

/// Given a point_id, returns the list of indices in the payload matching the condition
pub type NestedMatchingIndicesFn<'a> = Box<dyn Fn(PointOffsetType) -> Vec<ElemIndex> + 'a>;

/// Merge several nested condition results into a single regular condition checker
///
/// return a single condition checker that will return true if all nested condition checkers for the point_id
pub fn merge_nested_matching_indices(
    nested_checkers: Vec<NestedMatchingIndicesFn>,
) -> ConditionCheckerFn {
    Box::new(move |point_id: PointOffsetType| {
        // number of nested conditions to match
        let condition_count = nested_checkers.len();
        // binds payload `index` element to the number of matches it has accumulated
        let mut matches: HashMap<ElemIndex, usize> = HashMap::new();
        for nested_checker in &nested_checkers {
            let matching_indices = nested_checker(point_id);
            for index in matching_indices {
                let count = matches.entry(index).or_insert(0);
                *count += 1;
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
) -> NestedMatchingIndicesFn<'a> {
    match condition {
        Condition::Field(field_condition) => {
            // full path of the condition field
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
                            nested_check_field_condition(field_condition, &payload, nested_path)
                        })
                    })
                })
        }
        Condition::IsEmpty(is_empty) => Box::new(move |point_id| {
            payload_provider.with_payload(point_id, |payload| {
                check_nested_is_empty_condition(nested_path, is_empty, &payload)
            })
        }),
        Condition::IsNull(is_null) => Box::new(move |point_id| {
            payload_provider.with_payload(point_id, |payload| {
                check_nested_is_null_condition(nested_path, is_null, &payload)
            })
        }),
        Condition::HasId(_) => unreachable!(), // Is there a use case for this?
        Condition::Nested(_) => unreachable!(),
        Condition::Filter(_) => unreachable!(),
    }
}

/// Returns a checker function that will return the index of the payload elements
/// matching the condition for the given point_id
pub fn nested_field_condition_index<'a>(
    index: &'a FieldIndex,
    field_condition: &FieldCondition,
) -> Option<NestedMatchingIndicesFn<'a>> {
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
) -> Option<NestedMatchingIndicesFn> {
    match index {
        FieldIndex::GeoIndex(geo_index) => Some(Box::new(move |point_id: PointOffsetType| {
            match geo_index.get_values(point_id) {
                None => vec![],
                Some(values) => values
                    .iter()
                    .positions(|geo_point| geo_radius.check_point(geo_point.lon, geo_point.lat))
                    .collect(),
            }
        })),
        _ => None,
    }
}

pub fn get_nested_geo_bounding_box_checkers(
    index: &FieldIndex,
    geo_bounding_box: GeoBoundingBox,
) -> Option<NestedMatchingIndicesFn> {
    match index {
        FieldIndex::GeoIndex(geo_index) => Some(Box::new(move |point_id: PointOffsetType| {
            match geo_index.get_values(point_id) {
                None => vec![],
                Some(values) => values
                    .iter()
                    .positions(|geo_point| {
                        geo_bounding_box.check_point(geo_point.lon, geo_point.lat)
                    })
                    .collect(),
            }
        })),
        _ => None,
    }
}

pub fn get_nested_range_checkers(
    index: &FieldIndex,
    range: Range,
) -> Option<NestedMatchingIndicesFn> {
    match index {
        FieldIndex::IntIndex(num_index) => Some(Box::new(move |point_id: PointOffsetType| {
            match num_index.get_values(point_id) {
                None => vec![],
                Some(values) => values
                    .iter()
                    .copied()
                    .positions(|i| range.check_range(i as FloatPayloadType))
                    .collect(),
            }
        })),
        FieldIndex::FloatIndex(num_index) => Some(Box::new(move |point_id: PointOffsetType| {
            match num_index.get_values(point_id) {
                None => vec![],
                Some(values) => values
                    .iter()
                    .copied()
                    .positions(|i| range.check_range(i))
                    .collect(),
            }
        })),
        _ => None,
    }
}

pub fn get_nested_match_checkers(
    index: &FieldIndex,
    cond_match: Match,
) -> Option<NestedMatchingIndicesFn> {
    match cond_match {
        Match::Value(MatchValue {
            value: value_variant,
        }) => match (value_variant, index) {
            (ValueVariants::Keyword(keyword), FieldIndex::KeywordIndex(index)) => {
                Some(Box::new(move |point_id: PointOffsetType| {
                    match index.get_values(point_id) {
                        None => vec![],
                        Some(values) => values.iter().positions(|k| k == &keyword).collect(),
                    }
                }))
            }
            (ValueVariants::Integer(value), FieldIndex::IntMapIndex(index)) => {
                Some(Box::new(move |point_id: PointOffsetType| {
                    match index.get_values(point_id) {
                        None => vec![],
                        Some(values) => values.iter().positions(|i| i == &value).collect(),
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
                        None => vec![],
                        Some(doc) => {
                            let res = parsed_query.check_match(doc);
                            // Not sure it is entirely correct
                            if res {
                                vec![0]
                            } else {
                                vec![]
                            }
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
                        None => vec![],
                        Some(values) => values.iter().positions(|k| list.contains(k)).collect(),
                    }
                }))
            }
            (AnyVariants::Integers(list), FieldIndex::IntMapIndex(index)) => {
                Some(Box::new(move |point_id: PointOffsetType| {
                    match index.get_values(point_id) {
                        None => vec![],
                        Some(values) => values.iter().positions(|i| list.contains(i)).collect(),
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
    fn zero_matching_merge_nested_matching_indices() {
        let matching_indices_fn: Vec<NestedMatchingIndicesFn> = vec![
            Box::new(|_point_id: PointOffsetType| vec![]),
            Box::new(|_point_id: PointOffsetType| vec![]),
            Box::new(|_point_id: PointOffsetType| vec![]),
        ];

        let merged = merge_nested_matching_indices(matching_indices_fn);
        // none of the conditions are matching anything
        let result: bool = merged(0);
        assert!(!result);
    }

    #[test]
    fn single_matching_merge_merge_nested_matching_indices() {
        let matching_indices_fn: Vec<NestedMatchingIndicesFn> = vec![
            Box::new(|_point_id: PointOffsetType| vec![0]),
            Box::new(|_point_id: PointOffsetType| vec![0]),
            Box::new(|_point_id: PointOffsetType| vec![0]),
        ];

        let merged = merge_nested_matching_indices(matching_indices_fn);
        let result: bool = merged(0);
        assert!(result);
    }

    #[test]
    fn single_non_matching_merge_nested_matching_indices() {
        let matching_indices_fn: Vec<NestedMatchingIndicesFn> = vec![
            Box::new(|_point_id: PointOffsetType| vec![0]),
            Box::new(|_point_id: PointOffsetType| vec![0]),
            Box::new(|_point_id: PointOffsetType| vec![1]),
        ];
        let merged = merge_nested_matching_indices(matching_indices_fn);
        // does not because all the checkers are not matching the same path
        let result: bool = merged(0);
        assert!(!result);
    }

    #[test]
    fn many_matching_merge_nested_matching_indices() {
        let matching_indices_fn: Vec<NestedMatchingIndicesFn> = vec![
            Box::new(|_point_id: PointOffsetType| vec![0, 1]),
            Box::new(|_point_id: PointOffsetType| vec![0, 1]),
            Box::new(|_point_id: PointOffsetType| vec![0]),
        ];

        let merged = merge_nested_matching_indices(matching_indices_fn);
        // still matching because of the path '0' matches all conditions
        let result: bool = merged(0);
        assert!(result);
    }
}
