import argparse
import textwrap
from importlib import import_module

from kliff_torch import __version__

commands = [("dataset", "kliff_torch.cmdline.dataset"), ("model", "kliff_torch.cmdline.model")]


def main():
    parser = argparse.ArgumentParser(description="KLIFF utility collections.")
    parser.add_argument(
        "-v", "--version", action="version", version="KLIFF {}".format(__version__)
    )

    # sub-command
    subparsers = parser.add_subparsers(title="Sub-commands", dest="command")

    # help
    subparser = subparsers.add_parser(
        "help", description="Help", help="Help for sub-command."
    )
    subparser.add_argument(
        "helpcommand",
        nargs="?",
        metavar="sub-command",
        help="Provide help for sub-command.",
    )
    # add all subparser
    parsers = dict()
    functions = dict()
    for cmd_name, mod_name in commands:
        command = import_module(mod_name).Command
        docstring = command.__doc__
        parts = docstring.split("\n", 1)
        if len(parts) == 1:
            short = long = textwrap.dedent(docstring)
        else:
            short, body = parts
            long = short + "\n" + textwrap.dedent(body)

        subparser = subparsers.add_parser(cmd_name, help=short, description=long)
        command.add_arguments(subparser)
        parsers[cmd_name] = subparser
        functions[cmd_name] = command.run

    args = parser.parse_args()
    cmd_name = args.command

    if cmd_name is None:
        parser.print_help()
    elif cmd_name == "help":
        if args.helpcommand is None:
            parser.print_help()
        else:
            parsers[args.helpcommand].print_help()
    else:
        run = functions[cmd_name]
        try:
            if run.__code__.co_argcount == 1:
                run(args)
            else:
                run(args, parsers[cmd_name])
        except KeyboardInterrupt:
            pass
        except Exception as e:
            parser.error("{}: {}\n".format(e.__class__.__name__, e))


if __name__ == "__main__":
    main()
