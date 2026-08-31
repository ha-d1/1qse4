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


def fit_feature_encoder(train_rows, fields=None):
    fields = list(FIELDS if fields is None else fields)
    if fields != TEMPORAL_CROSS_FIELDS:
        return starter_data.fit_feature_encoder(train_rows, fields=fields)
    base_encoder = starter_data.fit_feature_encoder(train_rows, fields=SESSION_FIELDS)
    vocabs = [dict(), dict()]
    for values in _temporal_cross_values(train_rows, include_labels=True):
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
    if encoder.get("fields") != TEMPORAL_CROSS_FIELDS:
        return starter_data.transform_rows(rows, encoder, include_labels=include_labels)
    X_base, labels, users = starter_data.transform_rows(
        rows, encoder["base_encoder"], include_labels=include_labels
    )
    X = np.empty((len(rows), len(TEMPORAL_CROSS_FIELDS)), dtype=np.int32)
    X[:, : len(SESSION_FIELDS)] = X_base
    for row_index, values in enumerate(
        _temporal_cross_values(rows, include_labels=include_labels)
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
