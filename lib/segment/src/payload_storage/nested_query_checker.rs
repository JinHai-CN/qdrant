use std::collections::HashMap;
use std::ops::Deref;

use serde_json::Value;

use crate::common::utils::{JsonPathPayload, MultiValue};
use crate::payload_storage::condition_checker::ValueChecker;
use crate::types::{
    Condition, FieldCondition, Filter, IsEmptyCondition, IsNullCondition, OwnedPayloadRef, Payload,
};

fn check_all_nested_conditions<F>(checker: &F, must: &Option<Vec<Condition>>) -> bool
where
    F: Fn(&Condition) -> Vec<usize>,
{
    match must {
        None => true,
        Some(conditions) => {
            let condition_count = conditions.len();
            let matching_paths: Vec<usize> = conditions.iter().flat_map(checker).collect();
            // Count the number of matches per element index
            let mut matches: HashMap<usize, usize> = HashMap::new();
            for m in matching_paths {
                *matches.entry(m).or_insert(0) += 1;
            }
            matches.iter().any(|(_, count)| *count == condition_count)
        }
    }
}

pub fn check_nested_filter<'a, F>(
    nested_path: &JsonPathPayload,
    nested_filter: &Filter,
    get_payload: F,
) -> bool
where
    F: Fn() -> OwnedPayloadRef<'a>,
{
    let nested_checker = |condition: &Condition| match condition {
        Condition::Field(field_condition) => {
            nested_check_field_condition(field_condition, get_payload().deref(), nested_path)
        }
        Condition::IsEmpty(is_empty) => {
            check_nested_is_empty_condition(nested_path, is_empty, get_payload().deref())
        }
        Condition::IsNull(is_null) => {
            check_nested_is_null_condition(nested_path, is_null, get_payload().deref())
        }
        Condition::HasId(_) => unreachable!(), // Is there a use case for nested HasId?
        Condition::Nested(_) => unreachable!(), // Several layers of nesting are not supported here
        Condition::Filter(_) => unreachable!(),
    };

    nested_filter_checker(&nested_checker, nested_filter)
}

/// Warning only `must` conditions are supported for those tests
pub fn nested_filter_checker<F>(matching_paths: &F, nested_filter: &Filter) -> bool
where
    F: Fn(&Condition) -> Vec<usize>,
{
    // TODO add check_nested_should and check_nested_must_not
    check_all_nested_conditions(matching_paths, &nested_filter.must)
}

/// Return element indices matching the condition in the payload
pub fn check_nested_is_empty_condition(
    nested_path: &JsonPathPayload,
    is_empty: &IsEmptyCondition,
    payload: &Payload,
) -> Vec<usize> {
    // full nested path
    let full_path = nested_path.add_segment(&is_empty.is_empty.key);
    let field_values = payload.get_value(&full_path.path);

    let mut matching_indices = vec![];
    for (index, p) in field_values.values().iter().enumerate() {
        match p {
            Value::Null => matching_indices.push(index),
            Value::Array(vec) if vec.is_empty() => matching_indices.push(index),
            _ => (),
        }
    }
    matching_indices
}

/// Return element indices matching the condition in the payload
pub fn check_nested_is_null_condition(
    nested_path: &JsonPathPayload,
    is_null: &IsNullCondition,
    payload: &Payload,
) -> Vec<usize> {
    // full nested path
    let full_path = nested_path.add_segment(&is_null.is_null.key);
    let field_values = payload.get_value(&full_path.path);

    match field_values {
        MultiValue::Single(None) => vec![0],
        MultiValue::Single(Some(v)) => {
            if v.is_null() {
                vec![0]
            } else {
                vec![]
            }
        }
        MultiValue::Multiple(multiple_values) => {
            let mut paths = vec![];
            for (index, p) in multiple_values.iter().enumerate() {
                match p {
                    Value::Null => paths.push(index),
                    Value::Array(vec) => {
                        if vec.iter().any(|val| val.is_null()) {
                            paths.push(index)
                        }
                    }
                    _ => (),
                }
            }
            paths
        }
    }
}

/// Return indexes of the elements matching the condition in the payload values
pub fn nested_check_field_condition(
    field_condition: &FieldCondition,
    payload: &Payload,
    nested_path: &JsonPathPayload,
) -> Vec<usize> {
    let full_path = nested_path.add_segment(&field_condition.key);
    let field_values = payload.get_value(&full_path.path);

    let mut matching_indices = vec![];

    for (index, p) in field_values.values().iter().enumerate() {
        let mut res = false;
        // ToDo: Convert onto iterator over checkers, so it would be impossible to forget a condition
        res = res
            || field_condition
                .r#match
                .as_ref()
                .map_or(false, |condition| condition.check(p));
        res = res
            || field_condition
                .range
                .as_ref()
                .map_or(false, |condition| condition.check(p));
        res = res
            || field_condition
                .geo_radius
                .as_ref()
                .map_or(false, |condition| condition.check(p));
        res = res
            || field_condition
                .geo_bounding_box
                .as_ref()
                .map_or(false, |condition| condition.check(p));
        res = res
            || field_condition
                .values_count
                .as_ref()
                .map_or(false, |condition| condition.check(p));
        if res {
            matching_indices.push(index);
        }
    }
    matching_indices
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use atomic_refcell::AtomicRefCell;
    use serde_json::json;
    use tempfile::Builder;

    use super::*;
    use crate::common::rocksdb_wrapper::{open_db, DB_VECTOR_CF};
    use crate::id_tracker::simple_id_tracker::SimpleIdTracker;
    use crate::id_tracker::IdTracker;
    use crate::payload_storage::payload_storage_enum::PayloadStorageEnum;
    use crate::payload_storage::query_checker::SimpleConditionChecker;
    use crate::payload_storage::simple_payload_storage::SimplePayloadStorage;
    use crate::payload_storage::{ConditionChecker, PayloadStorage};
    use crate::types::{
        FieldCondition, GeoBoundingBox, GeoPoint, GeoRadius, PayloadField, Range, ValuesCount,
    };

    #[test]
    fn test_nested_condition_checker() {
        let dir = Builder::new().prefix("db_dir").tempdir().unwrap();
        let db = open_db(dir.path(), &[DB_VECTOR_CF]).unwrap();

        let payload_germany: Payload = json!(
        {
            "country": {
                "name": "Germany",
                "capital": "Berlin",
                "cities": [
                    {
                        "name": "Berlin",
                        "population": 3.7,
                        "location": {
                            "lon": 13.76116,
                            "lat": 52.33826,
                        },
                        "sightseeing": ["Brandenburg Gate", "Reichstag"]
                    },
                    {
                        "name": "Munich",
                        "population": 1.5,
                        "location": {
                            "lon": 11.57549,
                            "lat": 48.13743,
                        },
                        "sightseeing": ["Marienplatz", "Olympiapark"]
                    },
                    {
                        "name": "Hamburg",
                        "population": 1.8,
                        "location": {
                            "lon": 9.99368,
                            "lat": 53.55108,
                        },
                        "sightseeing": ["Reeperbahn", "Elbphilharmonie"]
                    }
                ],
            }
        })
        .into();

        let payload_japan: Payload = json!(
        {
            "country": {
                "name": "Japan",
                "capital": "Tokyo",
                "cities": [
                    {
                        "name": "Tokyo",
                        "population": 13.5,
                        "location": {
                            "lon": 139.69171,
                            "lat": 35.6895,
                        },
                        "sightseeing": ["Tokyo Tower", "Tokyo Skytree", "Tokyo Disneyland"]
                    },
                    {
                        "name": "Osaka",
                        "population": 2.7,
                        "location": {
                            "lon": 135.50217,
                            "lat": 34.69374,
                        },
                        "sightseeing": ["Osaka Castle", "Universal Studios Japan"]
                    },
                    {
                        "name": "Kyoto",
                        "population": 1.5,
                        "location": {
                            "lon": 135.76803,
                            "lat": 35.01163,
                        },
                        "sightseeing": ["Kiyomizu-dera", "Fushimi Inari-taisha"]
                    }
                ],
            }
        })
        .into();

        let payload_boring: Payload = json!(
        {
            "country": {
                "name": "Boring",
                "cities": [
                    {
                        "name": "Boring-ville",
                        "population": 0,
                        "sightseeing": [],
                    },
                ],
            }
        })
        .into();

        let mut payload_storage: PayloadStorageEnum =
            SimplePayloadStorage::open(db.clone()).unwrap().into();
        let mut id_tracker = SimpleIdTracker::open(db).unwrap();

        // point 0 - Germany
        id_tracker.set_link(0.into(), 0).unwrap();
        payload_storage.assign(0, &payload_germany).unwrap();

        // point 1 - Japan
        id_tracker.set_link(1.into(), 1).unwrap();
        payload_storage.assign(1, &payload_japan).unwrap();

        // point 2 - Boring
        id_tracker.set_link(2.into(), 2).unwrap();
        payload_storage.assign(2, &payload_boring).unwrap();

        let payload_checker = SimpleConditionChecker::new(
            Arc::new(AtomicRefCell::new(payload_storage)),
            Arc::new(AtomicRefCell::new(id_tracker)),
        );

        // single match condition in nested field no arrays
        let match_nested_name_condition = Filter::new_must(Condition::new_nested(
            "country".to_string(),
            Filter::new_must(Condition::Field(FieldCondition::new_match(
                "name".to_string(),
                "Germany".to_owned().into(),
            ))),
        ));

        assert!(payload_checker.check(0, &match_nested_name_condition));
        assert!(!payload_checker.check(1, &match_nested_name_condition));
        assert!(!payload_checker.check(2, &match_nested_name_condition));

        // single range condition nested field in array
        let population_range_condition = Filter::new_must(Condition::new_nested(
            "country.cities[]".to_string(),
            Filter::new_must(Condition::Field(FieldCondition::new_range(
                "population".to_string(),
                Range {
                    lt: None,
                    gt: Some(8.0),
                    gte: None,
                    lte: None,
                },
            ))),
        ));

        assert!(!payload_checker.check(0, &population_range_condition));
        assert!(payload_checker.check(1, &population_range_condition));
        assert!(!payload_checker.check(2, &population_range_condition));

        // single values_count condition nested field in array
        let sightseeing_value_count_condition = Filter::new_must(Condition::new_nested(
            "country.cities[]".to_string(),
            Filter::new_must(Condition::Field(FieldCondition::new_values_count(
                "sightseeing".to_string(),
                ValuesCount {
                    lt: None,
                    gt: None,
                    gte: Some(3),
                    lte: None,
                },
            ))),
        ));

        assert!(!payload_checker.check(0, &sightseeing_value_count_condition));
        assert!(payload_checker.check(1, &sightseeing_value_count_condition));
        assert!(!payload_checker.check(2, &sightseeing_value_count_condition));

        // single IsEmpty condition nested field in array
        let is_empty_condition = Filter::new_must(Condition::new_nested(
            "country.cities[]".to_string(),
            Filter::new_must(Condition::IsEmpty(IsEmptyCondition {
                is_empty: PayloadField {
                    key: "sightseeing".to_string(),
                },
            })),
        ));

        assert!(!payload_checker.check(0, &is_empty_condition));
        assert!(!payload_checker.check(1, &is_empty_condition));
        assert!(payload_checker.check(2, &is_empty_condition));

        // single IsNull condition nested field in array
        let is_empty_condition = Filter::new_must(Condition::new_nested(
            "country.cities[]".to_string(),
            Filter::new_must(Condition::IsNull(IsNullCondition {
                is_null: PayloadField {
                    key: "location".to_string(),
                },
            })),
        ));

        assert!(!payload_checker.check(0, &is_empty_condition));
        assert!(!payload_checker.check(1, &is_empty_condition));
        assert!(payload_checker.check(2, &is_empty_condition));

        // single geo-bounding box in nested field in array
        let location_close_to_berlin_box_condition = Filter::new_must(Condition::new_nested(
            "country.cities[]".to_string(),
            Filter::new_must(Condition::Field(FieldCondition::new_geo_bounding_box(
                "location".to_string(),
                GeoBoundingBox {
                    top_left: GeoPoint {
                        lon: 13.08835,
                        lat: 52.67551,
                    },
                    bottom_right: GeoPoint {
                        lon: 13.76117,
                        lat: 52.33825,
                    },
                },
            ))),
        ));

        // Germany has a city whose location is within the box
        assert!(payload_checker.check(0, &location_close_to_berlin_box_condition));
        assert!(!payload_checker.check(1, &location_close_to_berlin_box_condition));
        assert!(!payload_checker.check(2, &location_close_to_berlin_box_condition));

        // single geo-bounding box in nested field in array
        let location_close_to_berlin_radius_condition = Filter::new_must(Condition::new_nested(
            "country.cities[]".to_string(),
            Filter::new_must(Condition::Field(FieldCondition::new_geo_radius(
                "location".to_string(),
                GeoRadius {
                    center: GeoPoint {
                        lon: 13.76117,
                        lat: 52.33825,
                    },
                    radius: 1000.0,
                },
            ))),
        ));

        // Germany has a city whose location is within the radius
        assert!(payload_checker.check(0, &location_close_to_berlin_radius_condition));
        assert!(!payload_checker.check(1, &location_close_to_berlin_radius_condition));
        assert!(!payload_checker.check(2, &location_close_to_berlin_radius_condition));
    }
}
