# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from stats.lib.stats import corr

from afqmctools.utils.io import dump_dict

# Complex column pairs: maps complex identifier to (real_col, imag_col) tuple
COMPLEX_COLUMN_PAIRS = {
    'EnergyEstim': ('EnergyEstim__nume_real', 'EnergyEstim__nume_imag'),
    'EnergyDeno': ('EnergyEstim__deno_real', 'EnergyEstim__deno_imag'),
}

# Short aliases for common columns (user-friendly input)
COLUMN_ALIASES = {
    'energy': 'EnergyEstim',  # Now points to complex pair
    'energy_real': 'EnergyEstim__nume_real',
    'energy_imag': 'EnergyEstim__nume_imag',
    'deno': 'EnergyDeno',  # Complex denominator
    'deno_real': 'EnergyEstim__deno_real',
    'deno_imag': 'EnergyEstim__deno_imag',
    'energy_timer': 'EnergyEstim__timer',
    'pseudo': 'PseudoEloc',
    'overlap': 'LogOvlpFactor',
    'walkers': 'nWalkers',
    'memory': 'freeMemory',
    'eshift': 'Eshift',
    'shift': 'Eshift',
    'block': 'block',
    'time': 'time',
    'weight': 'weight',
    # Legacy compatibility
    'LocalEnergy': 'EnergyEstim',  # Also points to complex
}

# Pretty column names for display (long descriptive names)
PRETTY_COLUMN_NAMES = {
    'EnergyEstim': 'AFQMC Energy',
    'EnergyEstim__nume_real': 'AFQMC Energy (real)',
    'EnergyEstim__nume_imag': 'AFQMC Energy (imag)',
    'EnergyDeno': 'Energy Denominator',
    'EnergyEstim__deno_real': 'Energy Denominator (real)',
    'EnergyEstim__deno_imag': 'Energy Denominator (imag)',
    'EnergyEstim__timer': 'Energy Estimator Time',
    'PseudoEloc': 'Pseudo Local Energy',
    'LogOvlpFactor': 'Log Overlap Factor',
    'nWalkers': 'Number of Walkers',
    'freeMemory': 'Free Memory (MB)',
    'Eshift': 'Energy Shift',
    'block': 'Block',
    'time': 'Time',
    'weight': 'Walker Weight',
}

def is_complex_column(column_name):
    """
    Check if a column name refers to a complex column pair.
    
    Parameters
    ----------
    column_name : str
        Column identifier to check
    
    Returns
    -------
    bool
        True if this is a complex column with real/imag parts
    """
    return column_name in COMPLEX_COLUMN_PAIRS

def get_column_data(df, column_name):
    """
    Extract column data, handling both real and complex columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    column_name : str
        Column name (can be complex identifier or actual column)
    
    Returns
    -------
    np.ndarray
        Real array if single column, complex array if complex pair
    tuple or None
        (real_col_name, imag_col_name) if complex, None if real
    """
    if is_complex_column(column_name):
        real_col, imag_col = COMPLEX_COLUMN_PAIRS[column_name]
        data = df[real_col].values + 1j * df[imag_col].values
        return data, (real_col, imag_col)
    else:
        # Single real column
        return df[column_name].values, None

def perform_division(numerator, denominator):
    """
    Perform element-wise complex division.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array (can be real or complex)
    denominator : np.ndarray
        Denominator array (can be real or complex)
    
    Returns
    -------
    np.ndarray
        Result of numerator / denominator
    """
    return numerator / denominator

def resolve_column_name(user_input, df):
    """
    Resolve user input to actual column name or complex identifier.
    
    Accepts three formats:
    1. Short alias (e.g., 'energy', 'walkers')
    2. Actual column name (e.g., 'EnergyEstim__nume_real')
    3. Pretty display name (e.g., 'AFQMC Energy (real)')
    4. Complex column identifier (e.g., 'EnergyEstim')
    
    Parameters
    ----------
    user_input : str
        Column name/alias provided by user
    df : pd.DataFrame
        DataFrame to check for actual column names
    
    Returns
    -------
    str
        Actual column name, or complex column identifier
    
    Raises
    ------
    RuntimeError
        If the column cannot be resolved
    """
    # First, check if it's a complex column identifier
    if is_complex_column(user_input):
        real_col, imag_col = COMPLEX_COLUMN_PAIRS[user_input]
        if real_col in df.columns and imag_col in df.columns:
            return user_input
    
    # Second, check if it's already an actual column name
    if user_input in df.columns:
        return user_input
    
    # Third, check if it's a short alias
    if user_input in COLUMN_ALIASES:
        actual_name = COLUMN_ALIASES[user_input]
        # Check if it's a complex column
        if is_complex_column(actual_name):
            real_col, imag_col = COMPLEX_COLUMN_PAIRS[actual_name]
            if real_col in df.columns and imag_col in df.columns:
                return actual_name
        # Otherwise check if it's a regular column
        elif actual_name in df.columns:
            return actual_name
    
    # Fourth, check if it's a pretty name (reverse lookup)
    for actual_name, pretty_name in PRETTY_COLUMN_NAMES.items():
        if user_input == pretty_name:
            if is_complex_column(actual_name):
                real_col, imag_col = COMPLEX_COLUMN_PAIRS[actual_name]
                if real_col in df.columns and imag_col in df.columns:
                    return actual_name
            elif actual_name in df.columns:
                return actual_name
    
    # If we get here, the column was not found
    msg = f'Column "{user_input}" not found.\n'
    msg += f'Available columns: {list(df.columns)}\n'
    msg += f'Available aliases: {list(COLUMN_ALIASES.keys())}'
    raise RuntimeError(msg)

def read_scalar_table(args):
    fdat   = args.fname
    append = args.append

    if str(fdat).endswith('.csv'):
        return pd.read_csv(fdat)

    with open(fdat, 'r') as f:
        header = f.readline()
    if header.startswith('# column     1         -->'):    # i-PI output
        with open(fdat, 'r') as f:
            text = f.read()
        df = parse_output(text)
    else:    # generic scalar table file
        dfl = read_to_list(fdat, shebang=args.mark_header)
        df = pd.concat(dfl).reset_index(drop=True)

    # concatenate extra scalar.dat files
    if append is not None:
        dlist = [df]
        for fname in append:
            for df0 in read_to_list(fname, shebang=args.mark_header):
                dlist.append(df0)
        df = pd.concat(dlist, sort=False).reset_index()
    return df

def get_trace_figax():
    fig, ax_arr = plt.subplots(1, 2, sharey=True,
        gridspec_kw = {'width_ratios': [3, 1]}
    )
    return fig, ax_arr

def show_trace(ax, myx, myy0, nequil, ndiscard=None, **pyplot_kwargs):
    if ndiscard is None or ndiscard == 0:
        ax.plot(myx, myy0, c='k', label='')
        ax.axvline(nequil, c='k', ls='--', lw=2)
    elif isinstance(ndiscard,int):
        ax.plot(myx[ndiscard:], myy0[ndiscard:], c='k', label='')
        ax.axvline(nequil, c='k', ls='--', lw=2)

def show_histogram(ax, myy):
    ax.hist(myy, density=False, fc='gray', alpha=0.5,
        orientation='horizontal')

def overlay_statistics(ax, myy):
    mymean = myy.mean()
    mystd = myy.std(ddof=1)
    ax.axhline(mymean, c='b', lw=2, label="mean = %1.6f" % mymean)
    ax.axhline(mymean+mystd, ls=":", c="gray", lw=2,
        label="std         = %1.6f" % mystd)
    ax.axhline(mymean-mystd, ls=":", c="gray", lw=2)

def plot_trace(myx, myy0, nequil, xaxis, column,ndiscard=None,**pyplot_kwargs):
    sel = myx > nequil
    myy = myy0[sel]

    fig, ax_arr = get_trace_figax()

    if "fonsize" in pyplot_kwargs:
        fontsize = pyplot_kwargs["fonstize"]
    else:
        fontsize = 18

    # plot entire trace
    ax = ax_arr[0]
    ax.set_xlabel(xaxis, fontsize=fontsize)


    ax.set_ylabel(PRETTY_COLUMN_NAMES.get(column,column), fontsize=fontsize)
    show_trace(ax, myx, myy0, nequil, ndiscard=ndiscard, **pyplot_kwargs)


    if "xlims" in pyplot_kwargs:
        ax.set_xlim(*pyplot_kwargs["xlims"])

    # plot histogram of selected data
    ax = ax_arr[1]
    ax.set_xlabel('count', fontsize=fontsize)
    show_histogram(ax, myy)
    ax.get_yaxis().tick_right()

    # overlay statistics
    for ax in ax_arr:
        overlay_statistics(ax, myy)
    ax_arr[0].legend(loc='best')

    if "ylims" in pyplot_kwargs:
        ax.set_ylim(*pyplot_kwargs["ylims"])


    fig.tight_layout()
    return fig, ax_arr


def _get_default_args():
    """
    return default default arguments for analyze_scalar_data
    """
    return dict(
        fname = None,
        mark_header = "#",
        series_column = None,
        nequil = 0,
        estimate_equil = False,
        column = "LocalEnergy",
        reblock = 1,
        list = False,
        trace = False,
        append = None,
        dump = False,
        dump_fname = "trace.dat",
        savefig = None,
        ndiscard = None,
        return_data = False,
        autocorr = None,
        verbose = True,
        dump_avail_columns = False,
        denominator = None,
        return_complex = False
    )



def dict2namespace(in_dict):
    namespace = SimpleNamespace()
    for key,val in in_dict.items():
        setattr(namespace,key,val)
    return namespace


def analyze_scalar_data(args=None,**pyplot_kwargs):
    """
    Analyze scalar data from a scalar.dat file.

    Will calculate the mean and error of a single column of scalar data, and will automatically
    adjust for autocorrelation by default.

    Parameters
    ----------
    args : dict | SimpleNamespace | argparse.Namespace
        arguments for analyze_scalar_data. See notes below
        for details on possible keyword arguments and their
        respective defaults.
    pyplot_kwargs : dict
        keyword arguments for pyplot.

    Returns
    -------
    float
        mean of the scalar data
    float
        standard error of the mean of the scalar data (stddev / sqrt(n)) 
        where n is the number of *independent* samples

    Notes
    -----
    The following keyword arguments are available:

        
    Examples
    --------
    >>> analysis_settings = dict(
            fname = "qmc.s000.scalar.dat",
            xaxis = "time",    
            nequil = 5.0,       # length of equilibration phase in units of imaginary time (not steps!)
            trace = True,       # plots a trace of the scalar data
        )
        E,dE = analyze_scalar_data(analysis_settings)
    
    >>> analysis_settings = dict(
            fname = "qmc.s000.scalar.dat",
            nequil = 50,        # length of equilibration phase in units of MEASUREMENT BLOCKS
            trace = True,       # plots a trace of the scalar data
        )
        E,dE = analyze_scalar_data(analysis_settings)
    
    How to turn off autocorrelation correction:
    >>> analysis_settings = dict(
            fname = "qmc.s000.scalar.dat",
            nequil = 50,        # length of equilibration phase in units of MEASUREMENT BLOCKS
            trace = True,       # plots a trace of the scalar data,
            autocorr = 1    # sets the autocorrelation length to 1, effectively turning off autocorrelation correction
        )
        E,dE = analyze_scalar_data(analysis_settings)
    """

    if isinstance(args,dict):
        _args_dict = _get_default_args()

        for key in args:
            if key in _args_dict:
                _args_dict[key] = args[key]

        args = dict2namespace(_args_dict)

    if args.verbose and not args.dump_avail_columns:
        dump_dict(args.__dict__,dict_title= "analyze_scalar_data Settings")

    if args.list:
        with open(args.fname, 'r') as f:
            header = f.readline()
        if header.startswith(args.mark_header):
            cols = header[len(args.mark_header):].split()
            print(cols)
        else:
            print('no header')
        sys.exit(0)
    column = args.column
    if str(args.fname).endswith('.csv'):
        df = pd.read_csv(args.fname)
    else:    # read space-separated table
        df = read_scalar_table(args)
    # check or override "df" here

    if args.dump_avail_columns:
        print('\n====== [Available columns] ======\n')
        print(f'{"Column Name":<40s} {"Aliases":<30s} {"Pretty Name":<40s}')
        print('-' * 110)
        
        # Create reverse mapping of actual columns to their aliases
        for col in df.columns:
            # Find all aliases that point to this column
            aliases = [alias for alias, actual in COLUMN_ALIASES.items() if actual == col]
            aliases_str = ', '.join(aliases) if aliases else '-'
            pretty = PRETTY_COLUMN_NAMES.get(col, '-')
            
            # Wrap text for proper display
            col_lines = textwrap.wrap(col, width=38) or [col]
            alias_lines = textwrap.wrap(aliases_str, width=28) or [aliases_str]
            pretty_lines = textwrap.wrap(pretty, width=38) or [pretty]
            
            # Print first line
            max_lines = max(len(col_lines), len(alias_lines), len(pretty_lines))
            for i in range(max_lines):
                col_part = col_lines[i] if i < len(col_lines) else ''
                alias_part = alias_lines[i] if i < len(alias_lines) else ''
                pretty_part = pretty_lines[i] if i < len(pretty_lines) else ''
                print(f'{col_part:<40s} {alias_part:<30s} {pretty_part:<40s}')
        return None,None

    # interpret inputs and resolve column name
    if (column == 'LocalEnergy') and ('LocalEnergy' not in df.columns):
        # change default - now points to complex version
        column = 'EnergyEstim'
        # Fall back to first column if energy columns don't exist
        if not is_complex_column(column):
            if 'EnergyEstim__nume_real' in df.columns:
                column = 'EnergyEstim__nume_real'
            else:
                column = df.columns[0]
    
    # Resolve column name (supports aliases, actual names, and pretty names)
    column = resolve_column_name(column, df)
    
    # Handle denominator if specified
    denominator = args.denominator
    if denominator is not None:
        denominator = resolve_column_name(denominator, df)

    # calculate the mean and error of a single column of scalars
    if args.reblock > 1:
        df = reblock_scalar_df(df, args.reblock)
    
    # dump column if requested
    if args.dump:
        if is_complex_column(column):
            real_col, imag_col = COMPLEX_COLUMN_PAIRS[column]
            write(args.dump_fname, df[[real_col, imag_col]])
        else:
            write(args.dump_fname, df[[column]])

    # throw out equilibration
    series_column = args.series_column
    if series_column is None:
        myx = df["time"]
    elif series_column in ("block","time"):
        myx = df[series_column]
    else:
        raise ValueError(f"Invalid series_column: {series_column}. "
                         "Must be 'block' or 'time'.")
    nequil = args.nequil
    if args.estimate_equil:
        # Use real part for equilibration estimation
        if is_complex_column(column):
            real_col, _ = COMPLEX_COLUMN_PAIRS[column]
            nequil = nequil_std(df[real_col].values)
        else:
            nequil = nequil_std(df[column].values)
        nequil = myx.values[nequil]
    sel = myx > nequil
    
    # Extract data and handle complex columns
    is_complex = is_complex_column(column)
    
    if is_complex:
        # Get complex data
        data, (real_col, imag_col) = get_column_data(df.loc[sel], column)
        
        # Handle denominator division if specified
        if denominator is not None:
            denom_data, _ = get_column_data(df.loc[sel], denominator)
            data = perform_division(data, denom_data)
        
        # Analyze real part first to get correlation length
        ymean_real, yerr_real, ycorr = single_column_from_array(data.real, 0, kappa=args.autocorr)
        
        # Analyze imaginary part using same correlation length
        ymean_imag, yerr_imag, _ = single_column_from_array(data.imag, 0, kappa=ycorr)
        
        # For backwards compatibility, default return is real part only
        ymean, yerr = ymean_real, yerr_real
    else:
        # Single real column - original behavior
        data, _ = get_column_data(df.loc[sel], column)
        
        # Handle denominator division if specified
        if denominator is not None:
            denom_data, _ = get_column_data(df.loc[sel], denominator)
            data = perform_division(data, denom_data)
        
        ymean, yerr, ycorr = single_column_from_array(data.real if np.iscomplexobj(data) else data, 0, kappa=args.autocorr)
        ymean_real, yerr_real = ymean, yerr
        ymean_imag, yerr_imag = 0.0, 0.0

    if args.verbose:
        # print statistics
        pretty_name = PRETTY_COLUMN_NAMES.get(str(column), str(column))
        if is_complex:
            # Print both real and imaginary parts
            prt_format = "{name:40s} (real): {mean:10.6f} +/- {error:10.6f} {corr:4.2f} {nequil:4.1f}/{ndat:4.1f}"
            output = prt_format.format(
                name = pretty_name,
                mean = ymean_real,
                error= yerr_real,
                corr = ycorr,
                nequil = nequil,
                ndat = myx.max(),
            )
            print(output)
            prt_format = "{name:40s} (imag): {mean:10.6f} +/- {error:10.6f} {corr:4.2f}"
            output = prt_format.format(
                name = pretty_name,
                mean = ymean_imag,
                error= yerr_imag,
                corr = ycorr,
            )
            print(output)
        else:
            prt_format = "{name:40s} {mean:10.6f} +/- {error:10.6f} {corr:4.2f} {nequil:4.1f}/{ndat:4.1f}"
            output = prt_format.format(
                name = pretty_name,
                mean = ymean,
                error= yerr,
                corr = ycorr,
                nequil = nequil,
                ndat = myx.max(),
            )
            print(output)

    if args.trace:    # plot column
        ndiscard = getattr(args,"ndiscard",0)
        
        # Plot real part only
        if is_complex:
            real_col, _ = COMPLEX_COLUMN_PAIRS[column]
            myy0 = df[real_col].values
            # Apply denominator if specified
            if denominator is not None:
                full_data, _ = get_column_data(df, column)
                denom_full, _ = get_column_data(df, denominator)
                myy0 = perform_division(full_data, denom_full).real
        else:
            myy0 = df[column].values
            # Apply denominator if specified
            if denominator is not None:
                full_data, _ = get_column_data(df, column)
                denom_full, _ = get_column_data(df, denominator)
                myy0 = perform_division(full_data, denom_full).real if np.iscomplexobj(perform_division(full_data, denom_full)) else perform_division(full_data, denom_full)
        
        fig, ax_arr = plot_trace(myx, myy0, nequil, series_column, column, ndiscard=ndiscard, **pyplot_kwargs)
        if args.savefig:
            fig.savefig(args.savefig, dpi=320)
        else:
            plt.show()
    
    if hasattr(args,"return_data") and args.return_data:
        output = dict(
              name = str(column),
              mean_real = ymean_real,
              error_real = yerr_real,
              mean_imag = ymean_imag,
              error_imag = yerr_imag,
              corr = ycorr,
              nequil = nequil,
              ndat = myx.max(),
          )
        return output
    
    # Return value based on return_complex flag
    if args.return_complex and is_complex:
        return ymean_real + 1j*ymean_imag, yerr_real + 1j*yerr_imag
    else:
        # Backwards compatible: return real part only
        return ymean, yerr

def interpret_headers(headers):
    columns = []
    ncol = 0
    for header in headers:
        toks = header.split()
        ict = toks[2]
        assert toks[3] == '-->'
        col = toks[4]
        # append column names
        if '-' in ict:
            i, j = list(map(int, ict.split('-')))
            for ic in range(j-i+1):
                columns.append('%s_%d' % (col, ic))
        else:
            assert int(ict) == ncol+1
            columns.append(col)
        # update number of columns
        ncol = len(columns)
    return columns


def parse_output(text, ncol_max=1000):
    fp = get_string_io(text)
    # header lines describe columns
    headers = []
    for icol in range(ncol_max):
        header = fp.readline().strip()
        if header.startswith('# col'):
            headers.append(header)
        else:
            break
    # interpret column labels
    columns = interpret_headers(headers)
    df = pd.read_csv(fp, sep=r'\s+', header=None)
    fp.close()
    df.columns = columns
    return df


def reblock(trace, block_size, min_nblock=4, with_sigma=False):
  """ block scalar trace to remove autocorrelation;
  see usage example in reblock_scalar_df

  Args:
    trace (np.array): a trace of scalars, may have multiple columns
      !!!! assuming leading dimension is the number of current blocks.
    block_size (int): size of block in units of current block.
    min_nblock (int,optional): minimum number of blocks needed for
      meaningful statistics, default is 4.
  Returns:
    np.array: re-blocked trace.
  """
  nblock= len(trace)//block_size
  nkeep = nblock*block_size
  if (nblock < min_nblock):
    raise RuntimeError('only %d blocks left after reblock' % nblock)
  # end if
  blocked_trace = trace[:nkeep].reshape(nblock, block_size, *trace.shape[1:])
  ret = np.mean(blocked_trace, axis=1)
  if with_sigma:
    ret = (ret, np.std(blocked_trace, ddof=1, axis=1))
  return ret


def reblock_scalar_df(df, block_size, min_nblock=4):
  """ create a re-blocked scalar dataframe from a current scalar dataframe
   see reblock for details
  """
  return pd.DataFrame(
    reblock(df.values, block_size, min_nblock=min_nblock),
    columns=df.columns
  )


def write(dat_fname, df, header_pad='# ', end='\n', **kwargs):
    """Write dataframe to plain text scalar table format

    Lightly wrap around pandas.to_string with defaults to index and float_format

    Args:
        dat_fname (str): output data file name
        df (pd.DataFrame): data
        header_pad (str, optional): pad beginning of header with comment
         string, default '# '
    """
    default_kws = {
        'index': False,
        'float_format': '%12.8f'
    }
    for k, v in default_kws.items():
        if k not in kwargs:
            kwargs[k] = v
    text = df.to_string(**kwargs)
    with open(dat_fname, 'w') as f:
        f.write(header_pad + text + end)


def parse(text, shebang='#'):
    """Parse text of a scalar.dat file, should be table format.

    Args:
        text (str): content of scalar.dat file
        shebang (str, optional): marker for header line, default "#"
    Return:
        pd.DataFrame: table data
    Example:
        >>> with open('vmc.s001.scalar.dat', 'r') as f:
        >>>     text = f.read()
        >>>     df = parse(text)
    """
    fp = get_string_io(text)
    # try to read header line
    header = fp.readline().strip()
    fp.seek(0)
    # read data
    sep = r'\s+'
    if header.startswith(shebang):
        df = pd.read_csv(fp, sep=sep)
        # drop shebang from column names
        ncol_to_drop = len(shebang.split())
        columns = df.columns
        df.drop(columns[-ncol_to_drop:], axis=1, inplace=True)
        df.columns = columns[ncol_to_drop:]
        # calculate local energy variance if possible
        if ('LocalEnergy' in columns) and ('LocalEnergy_sq' in columns):
            df['Variance'] = df['LocalEnergy_sq']-df['LocalEnergy']**2.
    else:
        df = pd.read_csv(fp, sep=sep, header=None)
    fp.close()
    # column labels should be strings
    df.columns = map(str, df.columns)
    return df


def read_to_list(dat_fname, **kwargs):
    """Read scalar.dat file into a list of pandas DataFrames

    A line is a header if its first column cannot be converted to a float.
    Many scalar.dat files can be concatenated. A list will be returned.

    Args:
        dat_fname (str): name of input file
    Return:
        list: list of df(s) containing the table(s) of data
    Example:
        >>> dfl = read_to_list('gctas.dat')
        >>> df = pd.concat(dfl).reset_index(drop=True)
    """
    # first separate out the header lines and parse them
    with open(dat_fname, 'r') as f:
        text = f.read()
    idxl = find_header_lines(text)
    lines = text.split('\n')
    if len(idxl) == 0:    # no header
        return [parse(text, **kwargs)]
    idxl.append(None)
    # now read data and use headers to label columns
    lines = text.split('\n')
    dfl = []
    for bidx, eidx in zip(idxl[:-1], idxl[1:]):
        text1 = '\n'.join(lines[bidx:eidx])
        df1 = parse(text1, **kwargs)
        dfl.append(df1)
    return dfl


def get_string_io(text):
    """Obtain StringIO object from text

    Args:
        text (str): text to parse
    Return:
        StringIO: file-like object
    """
    
    from io import StringIO
    try:
        return StringIO(text.decode())
    except AttributeError:
        return StringIO(text)


def find_header_lines(text):
    """Find line numbers of all headers

    Args:
        text (str): text to parse
    Return:
        list: a list of integer line numbers
    """
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    fp = get_string_io(text)
    first_str = np.array(
        [is_float(line.split()[0]) for line in fp], dtype=bool)
    fp.close()
    idxl = np.where(~first_str)[0]
    return idxl.tolist()


def error(trace, kappa=None):
    """Calculate the error of a trace of scalar data

    Args:
        trace (list): should be a 1D iterable array of floating point numbers
        kappa (float,optional): auto-correlation time, default is to re-calculate
    Return:
        float: stderr, the error of the mean of this trace of scalars
    """
    stddev = np.std(trace, ddof=1)
    if np.isclose(stddev, 0):    # easy case
        return 0.0    # no error for constant trace
    if kappa is None:    # no call to corr
        kappa = corr(trace)
    neffective = np.sqrt(len(trace)/kappa)
    err = stddev/neffective
    return err


def single_column(df, column, nequil, kappa=None):
    """Calculate mean and error of a column

        nequil blocks of data are thrown out; autocorrelation time is taken into
    account when calculating error. The equilibrated data is assumed to have
    Gaussian distribution. Error is calculated for one standard deviation
    (1-sigma error).

    Args:
        df (pd.DataFrame): table of data (e.g. from parse)
        column (str): name of column
        nequil (int): number of equilibration blocks
        kappa (float, optional): autocorrelation time, default is to re-calculate
    Return:
        (float,float,float): (ymean,yerr,ycorr), where ymean is the mean of column
         , yerr is the 1-sigma error of column, and ycorr is the autocorrelation
    """
    myy = df[column].values[nequil:]
    return single_column_from_array(myy, nequil=0, kappa=kappa)

def single_column_from_array(data_array, nequil, kappa=None):
    """Calculate mean and error from a numpy array

        nequil data points are thrown out; autocorrelation time is taken into
    account when calculating error. The equilibrated data is assumed to have
    Gaussian distribution. Error is calculated for one standard deviation
    (1-sigma error).

    Args:
        data_array (np.ndarray): array of data
        nequil (int): number of equilibration points (usually 0 if already equilibrated)
        kappa (float, optional): autocorrelation time, default is to re-calculate
    Return:
        (float,float,float): (ymean,yerr,ycorr), where ymean is the mean,
         yerr is the 1-sigma error, and ycorr is the autocorrelation
    """
    myy = data_array[nequil:]

    ymean = np.mean(myy)
    if kappa is not None:
        ycorr = kappa
        yerr  = error(myy, ycorr)
    # check if data is uniform - based on absolute tolerance only since
    #  we want to know if the data is constant
    elif np.allclose(ymean*np.ones_like(myy), myy, rtol=0):
        print('uniform data detected')
        ycorr = np.inf
        yerr = 0.0
    else:
        ycorr = corr(myy)
        yerr  = error(myy, ycorr)
    return ymean, yerr, ycorr


def nequil_std(y, conv_frac=0.25, nsig=1, ntol=3):
    """Estimate number of equilibration blocks based on standard deviation

    Args:
        y (list): should be a 1D iterable array of floating point numbers
        conv_frac (float, optional): final fraction of trace that's assumed
            to be converged, default 0.25
        nsig (float, optional): number of sigmas (standard deviation) for a
            sample to be not equilibrated, default 3
        ntol (int, optional): number of consecutive samples to deviate for a
            trace to be not equilibrated, default 3
    Return:
        int: number of equilibration blocks
    """
    x = np.arange(len(y))
    ndat = len(x)
    # mean and stddev of converged portion
    nconv = int(round(ndat*conv_frac))
    if nconv < 4:
        msg = '%d/%d points is not enough' % (nconv, ndat)
        raise RuntimeError(msg)
    yc = y[:ndat-nconv]
    ym = np.mean(yc)
    ysig = np.std(yc, ddof=1)
    # find first block of points that exceeds nsig*ysig
    ytol = nsig*ysig
    nequil = ndat-nconv
    ngood = 0
    for y1 in y[ndat-nconv:0:-1]:
        dy = abs(y1-ym)
        if dy > ytol:
            ngood += 1
        else:
            ngood = 0
        if ngood >= ntol:
            break
        nequil -= 1
    nequil += int(round(ntol*3))
    return nequil
