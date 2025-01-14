# Vis script

Script to manage vis stock in a covenant. From a list of vis transaction (e.g. vis sources, ritual castings,
enchantments, ...) produces a ledger that describe the state of the vis stocks for all years and the consumption of each
magi, as well a detailed records of vis movement.

- [Vis script](#vis-script)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Examples](#examples)
    - [Inputs](#inputs)
      - [JSON Inputs (default)](#json-inputs-default)
      - [CSV Inputs](#csv-inputs)
    - [Outputs](#outputs)
      - [JSON](#json)
      - [Markdown](#markdown)

## Installation

The script is a `python>=3.12` script. We recommend using [`uv`][uv] to run the script, as `uv run --script vis.py` will
automatically manage python version and the dependencies needed to run the script.

The script has a shebang using `uv` to run it so that execution does not require managing a python virtual environment
and dependencies manually.

1. Install [`uv`][uv]
2. Download and copy the script somewhere
3. Depending on your operating system:
   - **Unix** (Linux, MacOS)
     1. Make the script executable with `chmod u+x vis.py`
     2. Run the script directly, e.g. `./vis.py --help`
   - **Windows** Windows does not understand shebang, so you'll have to invoke `uv` manually every time:
     `uv run --script vis.py`

The script can also be run directly with python >=3.12 with `python vis.py`. You are responsible for installing any
dependencies defined in the script [inline metadata][py-inline-metadata].

[uv]: https://docs.astral.sh/uv/
[py-inline-metadata]: https://packaging.python.org/en/latest/specifications/inline-script-metadata/

## Usage

The script follows POSIX semantics and supports reading from STDIN and writing to STDOUT/STDERR. Messages for the user
(e.g. logs) are written to STDERR, while machine-readable output (e.g. JSON) are written to STDOUT.

The `-i` and `-o` command-line arguments can be used to specify files to read from/write to. If `-i` and/or `-o` are not
passed, the script will by default read/write from/to STDIN/STDOUT. By default, the scripts reads and write:

- JSON if `-i` and/or `-o` are not provided
- else the format is determined from the file extension(s).

The `-input-format` and `--output-format` argument can be used to control the formats.

See also the `--help` message of the script for detailed CLI arguments.

The helper script `vis.sh` is a useful bash wrapper around the python script that:

- downloads & runs the latest version of the python code directly from github
- validate that the generated file has not been manually edited by maintaining a sha256 hash of the generated file

It assumes the input is named `vis.json` and the output `vis.md`. You can customize the variable in the file to reflect
your own setup.

### Examples

See the [examples/](./examples/) folder for example input and output files.

### Inputs

The inputs processed by the script are:

- A list of transactions, that is, changes to the stocks
- An optional starting stock

The supported input formats are either JSON or CSV.

#### JSON Inputs (default)

The input must be a JSON object with the following [cuelang][cuelang] schema:

```cuelang
records!: [#VisRecord]
start_stock?: #Vis

#VisRecord: {
    description: string
    vis: #Vis
    year: int
    season: "spring" | "summer" | "autumn" | "winter"
    magus?: string
    period?: int
    end_year?: int
}

#Vis: [string]: int

```

where:

- `records`: an array of transaction records, which describe input or output of vis in the stocks, like in a ledger. A
  record is made of:
  - `description`: A description of the event, such as a vis source name, the enchantment being made with the consumed
    vis, the ritual casted, ...
  - `vis`: What kind of vis is changed by what amount, e.g. +3He or -5 Vim. This is a mapping from Vis name to diff (so
    positive adds to the stock, negative removes from the stock). Vis name are just strings and the script will not
    unify abbreviated names (e.g. Im) with full names (e.g. Imaginem).
    - There is special handling for dedicated vis. You can use any of the `.,;:/|+&-` character as a separator, and this
      will be treated as dedicated vis (this affect sorting order of columns in markdown output). E.g. `Cr/Co` is Creo
      Corpus dedicated vis.
    - A single record can modify several type of vis at the same time since this entry is a JSON object from vis name to
      the amount change.
  - `year`: The year the record affects the stocks
  - `season`: The season the record affect the stocks. Must be one of `spring`, `summer`, `autumn`, `winter`
  - `magus` (optional): Name of the magus that brought or took the Vis from the stock
  - `period` (optional): For repeating records (e.g. an annual vis source, the _Aegis of the Hearth_ casting), the
    period in years of the repetitions. E.g. 1 for a modification to apply each year, 2 to apply every other year,
    etc...
  - `end_year` (optional): If `period` is define, the repeating input/output stops past this year
- `start_stock` (optional): The starting stock of vis available at the start of the accounting. Again, a JSON object
  mapping vis names to the starting amount.

[cuelang]: https://cuelang.org/docs/tour/

#### CSV Inputs

The CSV inputs is a list of record, one per row, with the same properties as in the JSON inputs. As CSV cannot support
the starting stock and several vis on the same row:

- the `vis` property is split into two columns, `art` and `amount`. The art of the vis (a string) must be written in the
  art column, and the amount of change in the `amount` column
- Rows which are identical save for the `art` and `amount` columns are aggregated together to form as single record,
  each line effectively specifying one entry in the `vis` mapping.
- If the `year` column is empty, the `art` and `amount` are added to the starting stock, and all other columns ignored

### Outputs

The script produces a ledger with detailed vis stocks as a function of time (year and season), total consumption for
each magi and detailed records of what happened to the stocks for each year and season.

#### JSON

The default output format is JSON, which contains full details. This include the recorded transaction and resulting
stocks for each seasons, as well as per-magus consumption tracking.

#### Markdown

The script can also produce a markdown report that is more human readable, featuring tables. In this case, the main
stock tables only describe the state of the stock per year, instead of per season, to keep it readable.
