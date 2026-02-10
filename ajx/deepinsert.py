import copy
import jax
import jax.numpy as jnp


def _is_dict(a):
    return isinstance(a, dict)


def _is_struct(a):
    return hasattr(a, "_flax_dataclass")


def _is_dictable(a):
    return _is_dict(a) or _is_struct(a) or hasattr(a, "as_dict")


def _as_dict(a):
    if _is_dict(a):
        return a
    if _is_struct(a):
        return a.__dict__
    if hasattr(a, "as_dict"):
        return a.as_dict()
    raise Exception("The provided source cannot be converted to a dict")


def _dict_array_insert(arr, index_dict):
    for key, value in index_dict.items():
        if isinstance(key, str):
            if key.isnumeric():
                key = int(key)
        if _is_dict(value):
            value = _dict_array_insert(arr[key], value)
            arr = arr.at[key].set(value)
        else:
            arr = arr.at[key].set(value)
    return arr


def deep_key_replace(dst, replace_map):
    dst_dict = _as_dict(dst)
    new_dst = copy.deepcopy(dst_dict)
    for old_key, value in dst_dict.items():
        if _is_dict(value):
            new_dst[old_key] = deep_key_replace(value, replace_map)

        if old_key in replace_map:
            new_key = replace_map[old_key]
            new_dst[new_key] = new_dst.pop(old_key)
    return new_dst


def deepinsert(dst, src):
    new = copy.deepcopy(dst)
    new_dict = _as_dict(new)
    src = _as_dict(src)
    for key in src.keys():
        if key in new_dict:
            if _is_struct(new_dict[key]):
                struct_type = type(new_dict[key])
                dst_dict = new_dict[key].__dict__
                dst_dict = deepinsert(dst_dict, src[key])
                new_dict[key] = struct_type(*dst_dict.values())
            elif _is_dictable(new_dict[key]):
                new_dict[key] = deepinsert(new_dict[key], src[key])
            elif _is_dict(src[key]):
                new_dict[key] = _dict_array_insert(new_dict[key], src[key])
            else:
                new_dict[key] = src[key]
        else:
            if "data" in new_dict:
                new_dict["data"] = _dict_array_insert(new_dict["data"], src)
            else:
                msg = (
                    f"The provided source ({key}) does not index destination correctly"
                )
                raise Exception(msg)
    return new


def deepindex(dst, src):
    if not isinstance(src, dict):
        raise Exception(
            "The source does not correspond to the tree structure of the destination"
        )
    for key in src.keys():
        if not isinstance(dst, dict):
            if _is_dict(src[key]):
                # Assumed that we are trying to index an nd array
                arr = dst[key]
                index_dict = src[key]
                return deepindex(arr, index_dict)
            else:
                return dst[key]
        elif key in dst:
            if _is_struct(dst[key]):
                return deepindex(dst[key].__dict__, src[key])
            elif _is_dict(dst[key]):
                return deepindex(dst[key], src[key])
            elif _is_dict(src[key]):
                # Assumed that we are trying to index an nd array
                arr = dst[key]
                index_dict = src[key]
                return deepindex(arr, index_dict)
            else:
                return dst[key]
    raise Exception("The provided source does not index destination correctly")


def flatten(p, keep_vectors=True, label=None):
    if _is_dictable(p):
        p = _as_dict(p)
        for k, v in p.items():
            if k == "data":
                yield from flatten(v, keep_vectors, "" if label is None else f"{label}")
                continue
            yield from flatten(v, keep_vectors, k if label is None else f"{label}.{k}")
    elif isinstance(p, tuple):
        for i, v in enumerate(p):
            yield from flatten(v, keep_vectors, i if label is None else f"{label}.{i}")
    elif _is_struct(p):
        for k, v in p.__dict__.items():
            yield from flatten(v, keep_vectors, k if label is None else f"{label}.{k}")
    elif isinstance(p, jax.Array):
        if len(p.shape) > 0:
            if keep_vectors:
                yield (label, p)
            else:
                for i in range(p.shape[0]):
                    yield from flatten(
                        p[i], keep_vectors, str(i) if label is None else f"{label}.{i}"
                    )
        else:
            yield (label, p)
    elif isinstance(p, list):
        if keep_vectors:
            yield (label, p)
        else:
            for i in range(len(p)):
                yield from flatten(
                    p[i], keep_vectors, str(i) if label is None else f"{label}.{i}"
                )
    else:
        yield (label, p)


def flatten_in_order(p, order, keep_vectors=True):
    flat = dict(flatten(p, keep_vectors))
    flat_ordered = {key: flat[key] for key in order}
    return flat_ordered


def pytree_as_vector(pytree, order, operation=None):
    flat = dict(flatten(pytree))
    if operation:
        vec = jnp.array([operation(flat[key]) for key in order]).flatten()
        return vec
    vec = jnp.array([flat[key] for key in order]).flatten()
    return vec


def vector_as_pytree(vector, keys, operation=None):
    if operation:
        flat_pytree = {key: operation(vector[i]) for i, key in enumerate(keys)}
        return unflatten(flat_pytree)
    flat_pytree = {key: vector[i] for i, key in enumerate(keys)}
    return unflatten(flat_pytree)


def merge(priority, destination):
    ddestination = _as_dict(destination)
    if not priority:
        return destination
    for key, value in priority.items():
        if isinstance(value, dict):
            node = ddestination.setdefault(key, {})
            merge(value, node)
        else:
            ddestination[key] = value

    return destination


def unflatten(p):
    result = {}
    for key, value in p.items():
        if "." in key:
            key_left, key_right = key.split(".", 1)
            if not key_left in result:
                result[key_left] = {}
            result[key_left] = merge(result[key_left], unflatten({key_right: value}))
        else:
            if key.isnumeric():
                key = int(key)
            result[key] = value
    return result


def get_pytree_len(pytree):
    flat_model_param, _ = jax.flatten_util.ravel_pytree(pytree)
    return len(flat_model_param)
