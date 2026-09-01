"""Candidate-only feature encoders built on the protected starter data contract."""
from __future__ import annotations

from datetime import date

import numpy as np

import data as starter_data

FIELDS = starter_data.FIELDS
WEEKDAY_FIELDS = starter_data.WEEKDAY_FIELDS
USER_AUTHOR_FIELDS = starter_data.USER_AUTHOR_FIELDS
HOUR_FIELDS = starter_data.HOUR_FIELDS
SESSION_FIELDS = starter_data.SESSION_FIELDS
TEMPORAL_CROSS_FIELDS = SESSION_FIELDS + ["weekday_hour", "hour_session_position"]
RECENT_SHORT_FIELDS = SESSION_FIELDS + [
    "author_seen_5",
    "video_seen_5",
    "author_distance_5",
]
RECENT_LONG_FIELDS = SESSION_FIELDS + [
    "author_seen_20",
    "video_seen_20",
    "author_distance_20",
]
RECENT_TIME_FIELDS = SESSION_FIELDS + [
    "author_recency_time",
    "video_recency_time",
    "author_frequency_10",
]
RECENT_FIELD_SETS = (RECENT_SHORT_FIELDS, RECENT_LONG_FIELDS, RECENT_TIME_FIELDS)


def _weekday(date_value) -> str:
    value = int(date_value)
    return str(date(value // 10000, (value // 100) % 100, value % 100).weekday())


def _temporal_cross_values(rows, include_labels):
    """Derive label-free time crosses while preserving the logged row order."""
    context_offset = 7 if include_labels else 6
    timestamp_offset = context_offset + 1
    last_timestamp = {}
    session_position = {}
    values = []
    for row in rows:
        if len(row) <= timestamp_offset:
            raise ValueError("temporal cross features require hourmin and time_ms context")
        user = row[1]
        timestamp = int(row[timestamp_offset])
        previous = last_timestamp.get(user)
        gap_ms = None if previous is None else timestamp - previous
        new_session = gap_ms is None or gap_ms < 0 or gap_ms > 30 * 60 * 1000
        position = 0 if new_session else session_position[user] + 1
        if position == 0:
            position_bucket = "0"
        elif position <= 2:
            position_bucket = "1-2"
        elif position <= 5:
            position_bucket = "3-5"
        elif position <= 10:
            position_bucket = "6-10"
        elif position <= 20:
            position_bucket = "11-20"
        else:
            position_bucket = "21+"
        hour = str(int(row[context_offset]) // 100)
        values.append((f"{_weekday(row[0])}:{hour}", f"{hour}:{position_bucket}"))
        last_timestamp[user] = timestamp
        session_position[user] = position
    return values


def _distance_bucket(distance, window):
    if distance is None or distance > window:
        return "unseen"
    if distance == 1:
        return "1"
    if distance <= 3:
        return "2-3"
    if distance <= 5:
        return "4-5"
    if distance <= 10:
        return "6-10"
    return "11-20"


def _time_bucket(delta_ms):
    if delta_ms is None or delta_ms < 0:
        return "unseen"
    if delta_ms <= 60 * 1000:
        return "0-1m"
    if delta_ms <= 5 * 60 * 1000:
        return "1-5m"
    if delta_ms <= 30 * 60 * 1000:
        return "5-30m"
    if delta_ms <= 6 * 60 * 60 * 1000:
        return "30m-6h"
    return "6h+"


def _recent_interest_values(rows, include_labels, fields):
    """Derive strictly prior, label-free exposure history in logged row order."""
    context_offset = 7 if include_labels else 6
    timestamp_offset = context_offset + 1
    recent = {}
    last_author_time = {}
    last_video_time = {}
    output = []
    window = 5 if fields == RECENT_SHORT_FIELDS else 20
    for row in rows:
        if len(row) <= timestamp_offset:
            raise ValueError("recent-interest features require hourmin and time_ms context")
        user, video, author = row[1], row[2], row[3]
        timestamp = int(row[timestamp_offset])
        history = recent.setdefault(user, [])
        author_distance = next(
            (index for index, item in enumerate(reversed(history), 1) if item[1] == author),
            None,
        )
        video_distance = next(
            (index for index, item in enumerate(reversed(history), 1) if item[0] == video),
            None,
        )
        if fields == RECENT_TIME_FIELDS:
            author_delta = (
                None
                if (user, author) not in last_author_time
                else timestamp - last_author_time[(user, author)]
            )
            video_delta = (
                None
                if (user, video) not in last_video_time
                else timestamp - last_video_time[(user, video)]
            )
            author_frequency = sum(item[1] == author for item in history[-10:])
            values = (
                _time_bucket(author_delta),
                _time_bucket(video_delta),
                "3+" if author_frequency >= 3 else str(author_frequency),
            )
        else:
            values = (
                "yes" if author_distance is not None and author_distance <= window else "no",
                "yes" if video_distance is not None and video_distance <= window else "no",
                _distance_bucket(author_distance, window),
            )
        output.append(values)
        history.append((video, author))
        if len(history) > 20:
            del history[:-20]
        last_author_time[(user, author)] = timestamp
        last_video_time[(user, video)] = timestamp
    return output


def _custom_values(rows, include_labels, fields):
    if fields == TEMPORAL_CROSS_FIELDS:
        return _temporal_cross_values(rows, include_labels=include_labels)
    if fields in RECENT_FIELD_SETS:
        return _recent_interest_values(rows, include_labels=include_labels, fields=fields)
    raise ValueError(f"Unsupported candidate feature set: {fields}")


def fit_feature_encoder(train_rows, fields=None):
    fields = list(FIELDS if fields is None else fields)
    if fields != TEMPORAL_CROSS_FIELDS and fields not in RECENT_FIELD_SETS:
        return starter_data.fit_feature_encoder(train_rows, fields=fields)
    base_encoder = starter_data.fit_feature_encoder(train_rows, fields=SESSION_FIELDS)
    vocabs = [dict() for _ in fields[len(SESSION_FIELDS) :]]
    for values in _custom_values(train_rows, include_labels=True, fields=fields):
        for index, value in enumerate(values):
            if value not in vocabs[index]:
                vocabs[index][value] = len(vocabs[index])
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum(
        [base_encoder["dimension"]] + dimensions[:-1]
    ).astype(np.int32)
    return {
        "fields": fields,
        "base_encoder": base_encoder,
        "cross_vocabs": vocabs,
        "cross_unknown": [len(vocab) for vocab in vocabs],
        "cross_offsets": offsets,
        "dimension": int(base_encoder["dimension"] + sum(dimensions)),
    }


def transform_rows(rows, encoder, include_labels=True):
    fields = encoder.get("fields")
    if fields != TEMPORAL_CROSS_FIELDS and fields not in RECENT_FIELD_SETS:
        return starter_data.transform_rows(rows, encoder, include_labels=include_labels)
    X_base, labels, users = starter_data.transform_rows(
        rows, encoder["base_encoder"], include_labels=include_labels
    )
    X = np.empty((len(rows), len(fields)), dtype=np.int32)
    X[:, : len(SESSION_FIELDS)] = X_base
    for row_index, values in enumerate(
        _custom_values(rows, include_labels=include_labels, fields=fields)
    ):
        for cross_index, value in enumerate(values):
            X[row_index, len(SESSION_FIELDS) + cross_index] = (
                encoder["cross_vocabs"][cross_index].get(
                    value, encoder["cross_unknown"][cross_index]
                )
                + encoder["cross_offsets"][cross_index]
            )
    return X, labels, users


def encode(splits, fields=None):
    if set(splits) != {"train", "valid"}:
        raise ValueError("Candidate encoding accepts train/valid only")
    encoder = fit_feature_encoder(splits["train"], fields=fields)
    encoded = {
        name: transform_rows(rows, encoder, include_labels=True)
        for name, rows in splits.items()
    }
    return encoded, encoder["dimension"]
