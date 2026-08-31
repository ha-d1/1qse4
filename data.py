"""KuaiRand-Pure data loading + official splits + feature encoding. Depends only on the standard library and numpy."""
import csv, os, collections
from datetime import date
from functools import lru_cache
import numpy as np

LABEL = 'long_view'
AUXILIARY_FIELDS = (
    'is_click',
    'is_like',
    'is_follow',
    'is_comment',
    'is_forward',
    'play_time_ms',
)
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 feature fields. Add features here — this is one of the main places students should modify.
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
WEEKDAY_FIELDS = FIELDS + ['weekday']
USER_AUTHOR_FIELDS = FIELDS + ['user_author']
HOUR_FIELDS = FIELDS + ['hour_bucket']
SESSION_FIELDS = HOUR_FIELDS + ['session_gap_bucket', 'session_position_bucket']
SUPPORTED_FIELDS = frozenset(
    WEEKDAY_FIELDS + USER_AUTHOR_FIELDS + HOUR_FIELDS + SESSION_FIELDS
)

def _load_selected(data_dir, split_names, include_labels=True):
    """Read only requested date ranges, optionally without materialising labels."""
    unknown = set(split_names) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    out = {name: [] for name in split_names}
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                date = int(r['date'])
                for name in split_names:
                    lo, hi = SPLITS[name]
                    if lo <= date <= hi:
                        row = (date, r['user_id'], r['video_id'],
                               vid2author.get(r['video_id'], 'UNK'), r['tab'],
                               float(r['duration_ms']))
                        if include_labels:
                            # Preserve the starter label at index 6; optional context follows it.
                            row += (1 if r[LABEL] != '0' else 0,
                                    int(r['hourmin']), int(r['time_ms']))
                        else:
                            row += (int(r['hourmin']), int(r['time_ms']))
                        out[name].append(row)
                        break
    return out


def load_selected(data_dir, split_names):
    """Load labels for an explicit subset of official splits."""
    return _load_selected(data_dir, tuple(split_names), include_labels=True)


def load_unlabelled(data_dir, split_name):
    """Load one split without reading its label column into returned rows."""
    return _load_selected(data_dir, (split_name,), include_labels=False)[split_name]


def load(data_dir):
    """Backward-compatible loader used by official baseline/submission utilities."""
    return load_selected(data_dir, SPLITS)


def load_train_auxiliary(data_dir, fields=AUXILIARY_FIELDS):
    """Load auxiliary targets for training dates only, aligned to train row order."""
    fields = tuple(fields)
    unknown = set(fields) - set(AUXILIARY_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported auxiliary fields: {sorted(unknown)}")
    values = {field: [] for field in fields}
    lo, hi = SPLITS['train']
    binary_fields = set(AUXILIARY_FIELDS) - {'play_time_ms'}
    for filename in (
        'log_standard_4_08_to_4_21_pure.csv',
        'log_standard_4_22_to_5_08_pure.csv',
    ):
        with open(os.path.join(data_dir, filename)) as fh:
            for row in csv.DictReader(fh):
                date_value = int(row['date'])
                if not lo <= date_value <= hi:
                    continue
                for field in fields:
                    if field in binary_fields:
                        values[field].append(1.0 if row[field] != '0' else 0.0)
                    else:
                        values[field].append(float(row[field]))
    return {
        field: np.asarray(field_values, dtype=np.float32)
        for field, field_values in values.items()
    }

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

@lru_cache(maxsize=None)
def _weekday(date_value):
    value = int(date_value)
    return str(date(value // 10000, (value // 100) % 100, value % 100).weekday())


def _validated_fields(fields=None):
    fields = list(FIELDS if fields is None else fields)
    unknown_fields = set(fields) - SUPPORTED_FIELDS
    if unknown_fields:
        raise ValueError(f"Unsupported feature fields: {sorted(unknown_fields)}")
    return fields


def _session_context(rows, include_labels):
    """Derive label-free within-user session state from event timestamps."""
    context_offset = 7 if include_labels else 6
    timestamp_offset = context_offset + 1
    last_timestamp = {}
    session_position = {}
    output = []
    for row in rows:
        if len(row) <= timestamp_offset:
            raise ValueError("session features require rows loaded with time_ms context")
        user = row[1]
        timestamp = int(row[timestamp_offset])
        previous = last_timestamp.get(user)
        gap_ms = None if previous is None else timestamp - previous
        new_session = gap_ms is None or gap_ms < 0 or gap_ms > 30 * 60 * 1000
        position = 0 if new_session else session_position[user] + 1
        if gap_ms is None:
            gap_bucket = "first"
        elif gap_ms < 0 or gap_ms > 30 * 60 * 1000:
            gap_bucket = "reset"
        elif gap_ms <= 10 * 1000:
            gap_bucket = "0-10s"
        elif gap_ms <= 60 * 1000:
            gap_bucket = "10-60s"
        elif gap_ms <= 5 * 60 * 1000:
            gap_bucket = "1-5m"
        else:
            gap_bucket = "5-30m"
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
        output.append(
            {
                "session_gap_bucket": gap_bucket,
                "session_position_bucket": position_bucket,
            }
        )
        last_timestamp[user] = timestamp
        session_position[user] = position
    return output


def _raw_features(row, fields, edges, include_labels=True, session_context=None):
    context_offset = 7 if include_labels else 6
    hourmin = row[context_offset] if len(row) > context_offset else None
    values = {
        'user_id': row[1],
        'video_id': row[2],
        'author_id': row[3],
        'tab': row[4],
        'dur_bucket': str(int(np.searchsorted(edges, row[5]))),
        'weekday': _weekday(row[0]),
        'user_author': f'{row[1]}\x1f{row[3]}',
    }
    if 'hour_bucket' in fields:
        if hourmin is None:
            raise ValueError("hour_bucket requires rows loaded with hourmin context")
        values['hour_bucket'] = str(int(hourmin) // 100)
    if 'session_gap_bucket' in fields or 'session_position_bucket' in fields:
        if session_context is None:
            raise ValueError("session fields require derived timestamp context")
        values.update(session_context)
    return [values[field] for field in fields]


def fit_feature_encoder(train_rows, fields=None):
    """Fit categorical vocabularies and duration buckets using training rows only."""
    fields = _validated_fields(fields)
    if not train_rows:
        raise ValueError("Cannot fit feature encoder without training rows")
    edges = _bucket_edges([row[5] for row in train_rows])
    vocabs = [dict() for _ in fields]
    contexts = (
        _session_context(train_rows, include_labels=True)
        if any(field.startswith('session_') for field in fields)
        else [None] * len(train_rows)
    )
    for row, context in zip(train_rows, contexts):
        for i, v in enumerate(
            _raw_features(
                row, fields, edges, include_labels=True, session_context=context
            )
        ):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    return {
        'fields': fields,
        'edges': edges,
        'vocabs': vocabs,
        'unk': unk,
        'field_dims': field_dims,
        'offsets': offsets,
        'dimension': int(sum(field_dims)),
    }


def transform_rows(rows, encoder, include_labels=True):
    """Apply a training-fitted encoder to labelled or unlabelled rows."""
    fields = encoder['fields']
    X = np.empty((len(rows), len(fields)), dtype=np.int32)
    y = np.empty(len(rows), dtype=np.float32) if include_labels else None
    users = []
    contexts = (
        _session_context(rows, include_labels=include_labels)
        if any(field.startswith('session_') for field in fields)
        else [None] * len(rows)
    )
    for n, (row, context) in enumerate(zip(rows, contexts)):
        for i, value in enumerate(
            _raw_features(
                row,
                fields,
                encoder['edges'],
                include_labels=include_labels,
                session_context=context,
            )
        ):
            X[n, i] = (
                encoder['vocabs'][i].get(value, encoder['unk'][i])
                + encoder['offsets'][i]
            )
        if include_labels:
            y[n] = row[6]
        users.append(row[1])
    return X, y, users


def encode(splits, fields=None):
    """Fit on train and encode labelled splits with train-only vocabularies."""
    encoder = fit_feature_encoder(splits['train'], fields=fields)
    encoded = {
        name: transform_rows(rows, encoder, include_labels=True)
        for name, rows in splits.items()
    }
    return encoded, encoder['dimension']
