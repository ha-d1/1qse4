import numpy as np
try:
    import lightgbm as _lgb
except ImportError:
    _lgb = None

from baseline import FM
from data import encode
from evaluate import evaluate


def _fit_fm_fixed(splits, k, lr, epochs, bs, seed):
    encoded, dimension = encode(splits)
    x_train, y_train, _ = encoded['train']
    model = FM(dimension, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), bs):
            batch = order[start:start + bs]
            model.step(x_train[batch], y_train[batch])
    return model, encoded


def _screen_fm(train_rows, holdout_rows, k, lr, max_epochs, bs, seed):
    screen_splits = {
        'train': train_rows,
        'valid': holdout_rows,
        'test': holdout_rows,
    }
    encoded, dimension = encode(screen_splits)
    x_train, y_train, _ = encoded['train']
    x_holdout, y_holdout, users_holdout = encoded['valid']
    model = FM(dimension, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary = float('-inf')
    best_epoch = max_epochs
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), bs):
            batch = order[start:start + bs]
            model.step(x_train[batch], y_train[batch])
        metrics = evaluate(users_holdout, y_holdout, model.predict(x_holdout))
        primary = float(metrics['primary'])
        if np.isfinite(primary) and primary > best_primary + 1e-5:
            best_primary = primary
            best_epoch = epoch
    return best_primary, best_epoch


def _fit_temporal_fm(splits, config, seed):
    """Select FM learning rate/epochs on a train-only temporal holdout."""
    train_rows = splits['train']
    early_rows = [row for row in train_rows if row[0] <= 20220418]
    holdout_rows = [row for row in train_rows if 20220419 <= row[0] <= 20220421]
    k = max(1, int(config.get('k', 16)))
    lr = float(config.get('lr', 0.001))
    epochs = max(1, int(config.get('epochs', 40)))
    bs = max(1, int(config.get('bs', 8192)))
    if not bool(config.get('temporal_select', False)):
        return _fit_fm_fixed(splits, k, lr, epochs, bs, seed)

    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError('lr must be finite and positive')
    if not early_rows or not holdout_rows:
        return _fit_fm_fixed(splits, k, lr, epochs, bs, seed)
    max_epochs = max(epochs, int(config.get('screen_epochs', epochs)))
    rates = (lr, lr * 0.5)
    best = (float('-inf'), lr, epochs)
    for candidate_lr in rates:
        primary, selected_epochs = _screen_fm(
            early_rows, holdout_rows, k, candidate_lr, max_epochs, bs, seed)
        if primary > best[0]:
            best = (primary, candidate_lr, selected_epochs)
    return _fit_fm_fixed(splits, k, best[1], best[2], bs, seed)


def _field_prior_adjustment(encoded, target_split, base):
    train_pack = encoded.get('train')
    target_pack = encoded.get(target_split)
    if train_pack is None or target_pack is None or len(train_pack) < 2:
        return base
    try:
        x_train = np.asarray(train_pack[0])
        y_train = np.asarray(train_pack[1], dtype=np.float64).reshape(-1)
        x_target = np.asarray(target_pack[0])
    except (TypeError, ValueError):
        return base
    if x_train.ndim != 2 or x_target.ndim != 2 or y_train.size != x_train.shape[0]:
        return base
    if x_target.shape[0] != base.size or x_train.shape[1] != x_target.shape[1]:
        return base
    if not np.all(np.isfinite(y_train)):
        return base
    rate = float(np.clip(np.mean(y_train), 1e-5, 1.0 - 1e-5))
    gl = np.log(rate / (1.0 - rate))
    adj = np.zeros(base.size, dtype=np.float64)
    used = 0
    for column in range(x_train.shape[1]):
        try:
            a = x_train[:, column]
            b = x_target[:, column]
            va = np.ones(a.size, dtype=bool)
            vb = np.ones(b.size, dtype=bool)
            if np.issubdtype(a.dtype, np.number):
                va &= np.isfinite(a)
                vb &= np.isfinite(b)
            keys, inv = np.unique(a[va], return_inverse=True)
            if keys.size == 0 or keys.size > 100000:
                continue
            cnt = np.bincount(inv, minlength=keys.size).astype(np.float64)
            pos = np.bincount(inv, weights=y_train[va], minlength=keys.size)
            loc = np.searchsorted(keys, b[vb])
            ok = loc < keys.size
            if keys.size:
                ok &= keys[np.minimum(loc, keys.size - 1)] == b[vb]
            rows = np.flatnonzero(vb)[ok]
            loc = loc[ok]
            posterior = (pos[loc] + 20.0 * rate) / (cnt[loc] + 20.0)
            adj[rows] += np.log(posterior / (1.0 - posterior)) - gl
            used += 1
        except (TypeError, ValueError, IndexError):
            continue
    return base if used == 0 else base + 0.35 * adj / float(used)


def _key(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return str(value)


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _value(row, name, index):
    if isinstance(row, dict):
        return row.get(name)
    if hasattr(row, '_asdict'):
        return row._asdict().get(name)
    try:
        return row[name]
    except (TypeError, KeyError, IndexError):
        try:
            return row[index]
        except (TypeError, IndexError):
            return None

def _context_history_features(data_access, target_split, x_train, x_target, y_train):
    """Build train-only numerical context and exposure-history features."""
    if data_access is None:
        return x_train, x_target, x_train.shape[1]
    columns = ('date', 'user_id', 'video_id', 'hourmin', 'time_ms',
               'duration_ms', 'tab', 'is_rand')
    train_file = 'log_standard_4_08_to_4_21_pure.csv'
    target_file = (train_file if target_split == 'train'
                   else 'log_standard_4_22_to_5_08_pure.csv')
    try:
        train_rows = list(data_access.iter_rows(train_file, columns, split='train'))
        target_rows = list(data_access.iter_rows(target_file, columns, split=target_split))
        video_rows = data_access.iter_rows(
            'video_features_basic_pure.csv', ('video_id', 'author_id'))
        author_by_video = {_key(_value(row, 'video_id', 0)): _key(
            _value(row, 'author_id', 1)) for row in video_rows}
    except Exception:
        return x_train, x_target, x_train.shape[1]
    if len(train_rows) != len(x_train) or len(target_rows) != len(x_target):
        return x_train, x_target, x_train.shape[1]
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if y_train.size != len(train_rows) or not np.isfinite(y_train).all():
        return x_train, x_target, x_train.shape[1]
    global_rate = float(np.clip(np.mean(y_train), 1e-5, 1.0 - 1e-5))
    prior = 20.0
    user_total, user_pos = {}, {}
    video_total, video_pos = {}, {}
    author_total, author_pos = {}, {}
    pair_total = {}
    last_user = {}

    def ordinal(row):
        date = _number(_value(row, 'date', 0))
        hourmin = _number(_value(row, 'hourmin', 3))
        if not np.isfinite(date):
            return np.nan
        minute = 0.0 if not np.isfinite(hourmin) else (
            np.floor(hourmin / 100.0) * 60.0 + hourmin % 100.0)
        return date * 1440.0 + minute

    def rate(total, positive):
        return (positive + prior * global_rate) / (total + prior)

    def build(rows, labels=None, update=False):
        features = np.zeros((len(rows), 12), dtype=np.float64)
        for index, row in enumerate(rows):
            user = _key(_value(row, 'user_id', 1))
            video = _key(_value(row, 'video_id', 2))
            author = author_by_video.get(video, video)
            pair = (user, video)
            total_u = user_total.get(user, 0.0)
            total_v = video_total.get(video, 0.0)
            total_a = author_total.get(author, 0.0)
            total_p = pair_total.get(pair, 0.0)
            time_value = ordinal(row)
            previous = last_user.get(user)
            gap = (max(0.0, time_value - previous)
                   if np.isfinite(time_value) and previous is not None else 0.0)
            duration = _number(_value(row, 'duration_ms', 5))
            hourmin = _number(_value(row, 'hourmin', 3))
            minute = 0.0 if not np.isfinite(hourmin) else (
                np.floor(hourmin / 100.0) * 60.0 + hourmin % 100.0)
            date = _number(_value(row, 'date', 0))
            random_flag = _number(_value(row, 'is_rand', 7))
            features[index] = (
                np.log1p(total_u),
                np.log1p(total_v),
                np.log1p(total_a),
                np.log1p(total_p),
                rate(total_u, user_pos.get(user, 0.0)),
                rate(total_v, video_pos.get(video, 0.0)),
                rate(total_a, author_pos.get(author, 0.0)),
                np.log1p(max(0.0, gap)),
                np.log1p(max(0.0, duration)) if np.isfinite(duration) else 0.0,
                minute / 1440.0,
                date - 20220408.0 if np.isfinite(date) else 0.0,
                random_flag if np.isfinite(random_flag) else 0.0,
            )
            if update:
                label = float(labels[index] > 0.5)
                user_total[user] = total_u + 1.0
                user_pos[user] = user_pos.get(user, 0.0) + label
                video_total[video] = total_v + 1.0
                video_pos[video] = video_pos.get(video, 0.0) + label
                author_total[author] = total_a + 1.0
                author_pos[author] = author_pos.get(author, 0.0) + label
                pair_total[pair] = total_p + 1.0
                if np.isfinite(time_value):
                    last_user[user] = time_value
        return features

    train_context = build(train_rows, y_train, update=True)
    target_context = build(target_rows)
    return (
        np.column_stack((np.asarray(x_train, dtype=np.float32), train_context)),
        np.column_stack((np.asarray(x_target, dtype=np.float32), target_context)),
        x_train.shape[1],
    )


def _watch_residual(data_access, target_split, n_target):
    columns = ('user_id', 'video_id', 'duration_ms', 'play_time_ms')
    train_file = 'log_standard_4_08_to_4_21_pure.csv'
    target_file = 'log_standard_4_08_to_4_21_pure.csv' if target_split == 'train' else 'log_standard_4_22_to_5_08_pure.csv'
    bounds = np.asarray([0.0, 10000.0, 20000.0, 40000.0, 80000.0, 120000.0, 180000.0, 260000.0, 400000.0, 700000.0, np.inf])
    sums = {}
    counts = {}
    bin_sum = np.zeros(10, dtype=np.float64)
    bin_count = np.zeros(10, dtype=np.float64)
    try:
        for row in data_access.iter_rows(train_file, columns, split='train'):
            duration = _number(_value(row, 'duration_ms', 2))
            play = _number(_value(row, 'play_time_ms', 3))
            video = _value(row, 'video_id', 1)
            if not np.isfinite(duration) or not np.isfinite(play) or duration <= 0.0 or video is None:
                continue
            bucket = int(np.searchsorted(bounds, duration, side='right') - 1)
            bucket = max(0, min(9, bucket))
            quality = float(np.clip(play / duration, 0.0, 1.5))
            key = _key(video)
            sums[key] = sums.get(key, 0.0) + quality
            counts[key] = counts.get(key, 0.0) + 1.0
            bin_sum[bucket] += quality
            bin_count[bucket] += 1.0
        global_mean = float(np.sum(bin_sum) / max(1.0, np.sum(bin_count)))
        bin_mean = (bin_sum + 30.0 * global_mean) / (bin_count + 30.0)
        item_residual = {}
        for key, total in sums.items():
            item_mean = (total + 30.0 * global_mean) / (counts[key] + 30.0)
            item_residual[key] = item_mean - global_mean
        output = np.zeros(n_target, dtype=np.float64)
        for index, row in enumerate(data_access.iter_rows(target_file, columns, split=target_split)):
            if index >= n_target:
                break
            duration = _number(_value(row, 'duration_ms', 2))
            video = _value(row, 'video_id', 1)
            if np.isfinite(duration) and duration > 0.0:
                bucket = int(np.searchsorted(bounds, duration, side='right') - 1)
                bucket = max(0, min(9, bucket))
                value = bin_mean[bucket] - global_mean
            else:
                value = 0.0
            if video is not None:
                value += item_residual.get(_key(video), 0.0)
            output[index] = value
        scale = float(np.std(output))
        if not np.isfinite(scale) or scale < 1e-6:
            return np.zeros(n_target, dtype=np.float64)
        return np.clip(output / scale, -3.0, 3.0)
    except Exception:
        return np.zeros(n_target, dtype=np.float64)


def _codes(matrix):
    array = np.asarray(matrix)
    if array.ndim != 2:
        return None
    if np.issubdtype(array.dtype, np.number):
        values = np.asarray(array, dtype=np.float64)
        values = np.nan_to_num(values, nan=-9.0e15, posinf=9.0e15, neginf=-9.0e15)
        return np.rint(values).astype(np.int64)
    result = np.empty(array.shape, dtype=np.int64)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            try:
                result[i, j] = int(float(array[i, j]))
            except (TypeError, ValueError):
                result[i, j] = hash(str(array[i, j]))
    return result


def _pairwise_delta(x_train, y_train, base_train, x_target, seed, config):
    x_train = np.asarray(x_train)
    x_target = np.asarray(x_target)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    base_train = np.asarray(base_train, dtype=np.float64).reshape(-1)
    if x_train.ndim != 2 or x_target.ndim != 2:
        return np.zeros(x_target.shape[0] if x_target.ndim else 0, dtype=np.float64)
    if x_train.shape[0] != y_train.size or x_train.shape[0] != base_train.size:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    if x_train.shape[1] != x_target.shape[1] or x_train.shape[0] < 4:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    codes_train = _codes(x_train)
    codes_target = _codes(x_target)
    if codes_train is None or codes_target is None:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    n, width = codes_train.shape
    if width > 1:
        max_cols = max(1, int(config.get('pair_max_cols', 32)))
        active = list(range(1, min(width, max_cols + 1)))
    else:
        active = [0]
    if not active:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    labels = np.isfinite(y_train) & (y_train > 0.5)
    users = codes_train[:, 0]
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    tables = [dict() for _ in active]
    rng = np.random.default_rng(int(seed) + 104729)
    epochs = max(1, int(config.get('pair_epochs', 3)))
    max_pairs = max(1, int(config.get('pair_max_pairs', 6)))
    learning_rate = float(config.get('pair_lr', 0.03))
    regularization = float(config.get('pair_reg', 0.001))
    groups = np.arange(starts.size)
    def adjusted_scores(ids):
        scores = np.asarray(base_train[ids], dtype=np.float64).copy()
        for table_index, column in enumerate(active):
            table = tables[table_index]
            scores += np.asarray(
                [table.get(int(key), 0.0) for key in codes_train[ids, column]],
                dtype=np.float64,
            )
        return scores

    for _ in range(epochs):
        rng.shuffle(groups)
        for group in groups:
            ids = order[starts[group]:ends[group]]
            positives = ids[labels[ids]]
            negatives = ids[~labels[ids]]
            if positives.size == 0 or negatives.size == 0:
                continue
            count = min(max_pairs, positives.size, negatives.size)
            if bool(config.get('hard_negatives', False)):
                positive_order = np.argsort(adjusted_scores(positives), kind='stable')
                negative_order = np.argsort(adjusted_scores(negatives), kind='stable')[::-1]
                positives = positives[positive_order[:count]]
                negatives = negatives[negative_order[:count]]
            else:
                positives = rng.choice(positives, size=count, replace=False)
                negatives = rng.choice(negatives, size=count, replace=False)

            for positive, negative in zip(positives, negatives):
                margin = float(base_train[positive] - base_train[negative])
                for table_index, column in enumerate(active):
                    pos_key = int(codes_train[positive, column])
                    neg_key = int(codes_train[negative, column])
                    if pos_key == neg_key:
                        continue
                    pos_value = tables[table_index].get(pos_key, 0.0)
                    neg_value = tables[table_index].get(neg_key, 0.0)
                    margin += pos_value - neg_value
                gradient = 1.0 / (1.0 + np.exp(np.clip(margin, -20.0, 20.0)))
                step = learning_rate * gradient
                for table_index, column in enumerate(active):
                    pos_key = int(codes_train[positive, column])
                    neg_key = int(codes_train[negative, column])
                    if pos_key == neg_key:
                        continue
                    table = tables[table_index]
                    pos_value = table.get(pos_key, 0.0)
                    neg_value = table.get(neg_key, 0.0)
                    table[pos_key] = pos_value + step - learning_rate * regularization * pos_value
                    table[neg_key] = neg_value - step - learning_rate * regularization * neg_value
    output = np.zeros(x_target.shape[0], dtype=np.float64)
    for table_index, column in enumerate(active):
        unique, inverse = np.unique(codes_target[:, column], return_inverse=True)
        values = np.asarray([tables[table_index].get(int(key), 0.0) for key in unique], dtype=np.float64)
        output += values[inverse]
    train_delta = np.zeros(n, dtype=np.float64)
    for table_index, column in enumerate(active):
        unique, inverse = np.unique(codes_train[:, column], return_inverse=True)
        values = np.asarray([tables[table_index].get(int(key), 0.0) for key in unique], dtype=np.float64)
        train_delta += values[inverse]
    scale = float(np.std(train_delta))
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    return np.clip(output / scale, -4.0, 4.0)


def _lightgbm_residual(x_train, y_train, base_train, x_target, seed, config,
                       categorical_count):
    """Fit a train-only LambdaRank correction on top of the current score."""
    if _lgb is None:
        return np.zeros(np.asarray(x_target).shape[0], dtype=np.float64)
    x_train = np.asarray(x_train)
    x_target = np.asarray(x_target)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    base_train = np.asarray(base_train, dtype=np.float64).reshape(-1)
    if x_train.ndim != 2 or x_target.ndim != 2:
        return np.zeros(x_target.shape[0] if x_target.ndim else 0, dtype=np.float64)
    if (x_train.shape[0] != y_train.size or x_train.shape[0] != base_train.size
            or x_train.shape[1] != x_target.shape[1] or x_train.shape[0] < 4):
        return np.zeros(x_target.shape[0], dtype=np.float64)
    if not np.isfinite(y_train).all() or not np.isfinite(base_train).all():
        return np.zeros(x_target.shape[0], dtype=np.float64)
    order = np.argsort(x_train[:, 0], kind='mergesort')
    ordered_users = x_train[order, 0]
    boundaries = np.flatnonzero(ordered_users[1:] != ordered_users[:-1]) + 1
    groups = np.diff(np.r_[0, boundaries, len(order)])
    rounds = max(1, int(config.get('lgb_rounds', 150)))
    learning_rate = float(config.get('lgb_lr', 0.05))
    leaves = max(2, int(config.get('lgb_num_leaves', 31)))
    min_data = max(1, int(config.get('lgb_min_data_in_leaf', 100)))
    lambda_l2 = max(0.0, float(config.get('lgb_lambda_l2', 1.0)))
    threads = max(1, int(config.get('lgb_threads', 4)))
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_at': [5],
        'learning_rate': learning_rate,
        'num_leaves': leaves,
        'min_data_in_leaf': min_data,
        'lambda_l2': lambda_l2,
        'verbosity': -1,
        'seed': int(seed),
        'feature_fraction_seed': int(seed),
        'bagging_seed': int(seed),
        'data_random_seed': int(seed),
        'num_threads': threads,
        'feature_pre_filter': False,
        'deterministic': True,
        'force_col_wise': True,
    }
    train_set = _lgb.Dataset(
        x_train[order],
        label=y_train[order],
        group=groups,
        init_score=base_train[order],
        categorical_feature=list(range(categorical_count)),
        free_raw_data=False,
    )
    booster = _lgb.train(params, train_set, num_boost_round=rounds)
    correction = np.asarray(booster.predict(x_target), dtype=np.float64).reshape(-1)
    return np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)


def score(splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    model, encoded = _fit_temporal_fm(splits, config, seed)
    x_train, y_train, _ = encoded['train']
    x_target, _, _ = encoded[target_split]
    base_train = np.asarray(model.predict(x_train), dtype=np.float64).reshape(-1)
    base_target = np.asarray(model.predict(x_target), dtype=np.float64).reshape(-1)
    base_train = np.nan_to_num(base_train, nan=0.0, posinf=30.0, neginf=-30.0)
    base_target = np.nan_to_num(base_target, nan=0.0, posinf=30.0, neginf=-30.0)
    residual = _field_prior_adjustment(encoded, target_split, base_target)
    watch = _watch_residual(data_access, target_split, base_target.size)
    pairwise = _pairwise_delta(x_train, y_train, base_train, x_target, seed, config)
    blend = float(config.get('watch_blend', 0.085))
    pair_blend = float(config.get('pair_blend', 0.8))
    result = residual + blend * watch + pair_blend * pairwise
    lightgbm_blend = float(config.get('lightgbm_blend', 0.5))
    if not np.isfinite(lightgbm_blend):
        raise ValueError('lightgbm_blend must be finite')
    if _lgb is not None and lightgbm_blend != 0.0:
        train_residual = _field_prior_adjustment(encoded, 'train', base_train)
        train_watch = _watch_residual(data_access, 'train', base_train.size)
        train_pairwise = _pairwise_delta(x_train, y_train, base_train, x_train, seed, config)
        current_train = train_residual + blend * train_watch + pair_blend * train_pairwise
        if bool(config.get('context_features', False)):
            lgb_train, lgb_target, categorical_count = _context_history_features(
                data_access, target_split, x_train, x_target, y_train)
        else:
            lgb_train, lgb_target, categorical_count = x_train, x_target, x_train.shape[1]
        result += lightgbm_blend * _lightgbm_residual(
            lgb_train, y_train, current_train, lgb_target, seed, config,
            categorical_count)
    return np.nan_to_num(result, nan=0.0, posinf=30.0, neginf=-30.0)
