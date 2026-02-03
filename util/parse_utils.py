from loguru import logger
import argparse
import sys
import os
import yaml

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"


def set_jax_settings(args):
    import jax

    jax.config.update("jax_compilation_cache_dir", ".jax-cache")
    jax.numpy.set_printoptions(threshold=jax.numpy.inf)
    jax.numpy.set_printoptions(edgeitems=30, linewidth=1000)

    if args.single_precision:
        jax.config.update("jax_enable_x64", False)
        logger.warning("Executing with single precision")
    else:
        jax.config.update("jax_enable_x64", True)

    if args.gpu:
        jax.config.update("jax_platform_name", "gpu")
    else:
        jax.config.update("jax_platform_name", "cpu")
    jax.devices()  # Jax forgets the device otherwise...
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        logger.warning("JIT compilation is disabled")


def parse_args(**override_args):
    logger.info("Importing JAX modules")
    # Slow imports are collected here
    import jax
    import jax.numpy
    import flax
    import jaxopt
    import xarray

    logger.info("Parsing arguments")

    filtered_priority_args = filter(lambda x: x.startswith("--"), sys.argv)
    priority_args = [argname[2:] for argname in filtered_priority_args]
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument("--no_overwrite", action="store_true")
    parser.add_argument("--debug_nans", action="store_true")
    parser.add_argument("--disable_jit", action="store_true")
    parser.add_argument("--single_precision", action="store_true")
    parser.add_argument("--gpu", action="store_true")

    parser.add_argument("--log_level_trace", action="store_true")
    parser.add_argument("--silent", action="store_true")

    parser.add_argument("--force_unsafe_save", action="store_true")
    parser.add_argument("--preprocess_only", action="store_true")
    parser.add_argument("--skip_optimization", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--do_free_run_mse", action="store_true")
    parser.add_argument("--save_residuals", action="store_true")
    parser.add_argument("--skip_explore", action="store_true")
    parser.add_argument("--load_fname", action="store", default="temp")
    parser.add_argument("--load_param", action="store_true")
    parser.add_argument("--load_ic", action="store_true")
    parser.add_argument("--only_optimize_ic", action="store_true")

    parser.add_argument("--fname", action="store", default="temp")
    parser.add_argument("--save", action="store", default="")
    parser.add_argument("--cycle", action="store", default="0")
    parser.add_argument("--conf", action="store", default="")

    parser.add_argument("--config_file", help="YAML configuration")
    parser.add_argument("--config_index", action="store")
    parser.add_argument(
        "--settings",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite settings (e.g. environment_settings:dict(timestep=0.02))",
        default=dict(),
    )
    args = parser.parse_args()
    has_protected_overwrite = False
    if not args.no_overwrite:
        for override, value in override_args.items():
            if not override in priority_args:
                # if args.__dict__[override] == parser.get_default
                args.__dict__[override] = value
            else:
                has_protected_overwrite = True

    args.cycle = int(args.cycle)

    if args.log_level_trace:
        logger.remove(0)
        logger.add(sys.stderr, level="TRACE")
        logger.trace("Log level is set to TRACE")
    if args.silent:
        logger.disable("")

    set_jax_settings(args)
    if has_protected_overwrite:
        logger.warning(
            f"Found command line arguments that overwrites parse_args arguments"
        )
    logger.info(
        f"command line arguments > parse_args arguments > config file arguments > default arguments"
    )
    logger.trace("Arguments have been parsed")

    return args


def safe_dict_update(dst, args):
    from util.deepinsert import deepinsert

    configuration = {}
    if args.config:
        with open(args.config, "r") as file:
            configuration = yaml.safe_load(file)
        if args.config_index:
            configuration = configuration[args.config_index]

    return deepinsert(dst, configuration)


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            arg_dict_local = self.split(arguments)
            arg_dict = {**arg_dict, **arg_dict_local}
        setattr(namespace, self.dest, arg_dict)

    def split(self, arguments):
        arg_dict = {}
        key = arguments.split(":")[0]
        value = ":".join(arguments.split(":")[1:])
        # Evaluate the string as python code
        try:
            if ":" in value:
                arg_dict_lower = self.split(value)
                arg_dict[key] = arg_dict_lower
            else:
                arg_dict[key] = eval(value)
        except NameError:
            arg_dict[key] = value
        except SyntaxError:
            return {key: value}

        return arg_dict
