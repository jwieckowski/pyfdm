# Copyright (c) 2026 Jakub Więckowski

import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

__all__ = ['StepLogger']

_VALID_FORMATS = ('console', 'json', 'csv', 'excel')

class StepLogger:
    """
    Collects named intermediate results from a fuzzy MCDA computation
    and renders them in one or more formats.

    Formats
    -------
    console
        Prints aligned tables to stdout.
    json
        Writes a structured JSON file.
    csv
        Writes one CSV file with all steps separated by blank rows.
    excel
        Writes an .xlsx workbook with one sheet per step.
        Requires openpyxl: ``pip install pyfdm[excel]``.

    Parameters
    ----------
        output : str or list of str
            One or more output formats. Each string must be one of:
            'console', 'json', 'csv', 'excel'.
            For file-based formats the path is derived from *path*.

        path : str or Path, optional
            Base path (without extension) for file outputs.
            Ignored for 'console'.  Defaults to 'pyfdm_steps_<timestamp>'.

        alternative_names : list of str, optional
            Row labels for matrix steps.

        criterion_names : list of str, optional
            Column labels for matrix steps.

        float_fmt : str, default '.4f'
            Format string for floating-point values.

    Examples
    --------
        >>> logger = StepLogger(output='console')
        >>> logger = StepLogger(output='json, path='results/topsis')
        >>> ftopsis = fTOPSIS(logger=logger)
        >>> prefs = ftopsis(matrix, weights, types)
        >>> logger.save()        # write file-based outputs
        >>> logger.print()       # force console output
    """

    def __init__(self, output='console', path=None,
                 alternative_names=None, criterion_names=None,
                 float_fmt='.4f'):
        self.output  = self._parse_output(output)
        self.path    = Path(path) if path else Path(
            f'pyfdm_steps_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.alt_names  = alternative_names
        self.crit_names = criterion_names
        self.float_fmt  = float_fmt

        self._steps: list[dict] = []   # ordered list of {name, data, shape_hint}
        self._method_name: str = ''

    #  Internal API
    def log(self, name: str, data, description: str = '') -> None:
        """
        Record one intermediate result.

        Parameters
        ----------
            name : str
                Human-readable step name, e.g. 'Normalised matrix'.

            data : ndarray or scalar or dict
                The intermediate value.  Arrays of ndim 1, 2, or 3 are
                supported.  Scalars and dicts are also accepted.

            description : str, optional
                One-line description shown in console/file headers.
        """
        self._steps.append({
            'name': name,
            'description': description,
            'data': data,
        })

    def _set_method(self, name: str) -> None:
        self._method_name = name

    def _resolve_labels(self, arr: np.ndarray):
        """Return (row_labels, col_labels) appropriate for *arr*.

        Row labels fall back to positional defaults when the stored
        alt_names / crit_names count does not match the array dimension,
        preventing IndexError when a method logs a criterion-indexed
        matrix (e.g. FPIS/FNIS of shape (n_criteria, 3)).
        """
        if arr.ndim == 1:
            m = arr.shape[0]
            rows = self.alt_names if (self.alt_names and len(self.alt_names) == m)                    else [f'A{i+1}' for i in range(m)]
            return rows, None
        if arr.ndim >= 2:
            m, n = arr.shape[0], arr.shape[1]
            rows = self.alt_names if (self.alt_names and len(self.alt_names) == m)                    else [f'R{i+1}' for i in range(m)]
            cols = self.crit_names if (self.crit_names and len(self.crit_names) == n)                    else [f'C{j+1}' for j in range(n)]
            return rows, cols
        return [], []

    # ------------------------------------------------------------------ #
    #  Public output methods                                               #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Write all file-based outputs (json / csv / excel)."""
        if 'json' in self.output:
            self._write_json()
        if 'csv' in self.output:
            self._write_csv()
        if 'excel' in self.output:
            self._write_excel()

    def print(self) -> None:
        """Print all steps to the console as formatted tables."""
        self._write_console()

    def clear(self) -> None:
        """Remove all recorded steps (useful for re-running a method)."""
        self._steps.clear()

    @property
    def steps(self) -> list:
        """Read-only list of recorded step dicts."""
        return list(self._steps)

    #  Console rendering                                                  
    def _write_console(self) -> None:
        width = 72
        header = f'  {self._method_name} — Intermediate Results  '
        print('\n' + '═' * width)
        print(header.center(width))
        print('═' * width)

        for idx, step in enumerate(self._steps, 1):
            name = step['name']
            desc = step['description']
            data = step['data']

            print(f'\n  [{idx}] {name}')
            if desc:
                print(f'      {desc}')
            print('  ' + '─' * (width - 2))

            self._print_data(data)

        print('\n' + '═' * width + '\n')

    def _print_data(self, data):

        if isinstance(data, dict):
            for k, v in data.items():
                print(f"\n{k}")
                self._print_data(v)
            return

        if not isinstance(data, np.ndarray):
            print(data)
            return

        arr = np.asarray(data)

        rows, cols = self._resolve_labels(arr)
        fmt = self.float_fmt

        if arr.ndim == 1:

            table = [arr.tolist()]

            print(
                tabulate(
                    table,
                    headers=rows,
                    tablefmt="rounded_outline",
                    floatfmt=fmt,
                )
            )

        elif arr.ndim == 2:
            table = []

            for label, row in zip(rows, arr):
                table.append([label] + row.tolist())

            print(
                tabulate(
                    table,
                    headers=[""] + cols,
                    tablefmt="rounded_outline",
                    floatfmt=fmt,
                )
            )

        elif arr.ndim == 3:
            table = []

            for rlabel, row in zip(rows, arr):

                values = []

                for l, m, u in row:
                    values.append(
                        f"[{l:{fmt}}, {m:{fmt}}, {u:{fmt}}]"
                    )

                table.append([rlabel] + values)

            print(
                tabulate(
                    table,
                    headers=[""] + cols,
                    tablefmt="rounded_outline",
                )
            )

    #  JSON                                                               
    def _write_json(self) -> None:
        path = self.path.with_suffix('.json')
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'method': self._method_name,
            'generated': datetime.now().isoformat(),
            'steps': []
        }

        for step in self._steps:
            entry = {
                'name': step['name'],
                'description': step['description'],
                'data': self._to_json_serialisable(step['data']),
            }
            payload['steps'].append(entry)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f'[StepLogger] JSON saved → {path}')

    def _to_json_serialisable(self, data):
        if isinstance(data, np.ndarray):
            arr = data.astype(float)
            rows, cols = self._resolve_labels(arr)
            if arr.ndim == 1:
                return {
                    'shape': list(arr.shape),
                    'labels': rows,
                    'values': arr.tolist(),
                }
            elif arr.ndim == 2:
                return {
                    'shape': list(arr.shape),
                    'row_labels': rows,
                    'col_labels': cols,
                    'values': arr.tolist(),
                }
            elif arr.ndim == 3:
                m, n, _ = arr.shape
                row_labels = rows or [f'A{i+1}' for i in range(m)]
                col_labels = cols or [f'C{j+1}' for j in range(n)]
                tfn_data = {}
                for i, rl in enumerate(row_labels):
                    tfn_data[rl] = {}
                    for j, cl in enumerate(col_labels):
                        tfn_data[rl][cl] = arr[i, j].tolist()
                return {'shape': list(arr.shape), 'tfn': tfn_data}
            return {'values': arr.tolist()}
        if isinstance(data, dict):
            return {k: self._to_json_serialisable(v) for k, v in data.items()}
        if isinstance(data, (float, int, np.floating, np.integer)):
            return float(data)
        return str(data)

    #  CSV                                                                 
    def _write_csv(self) -> None:
        path = self.path.with_suffix('.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = f'{{:{self.float_fmt}}}'

        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([f'# {self._method_name} — Intermediate Steps'])
            w.writerow([f'# Generated: {datetime.now().isoformat()}'])
            w.writerow([])

            for idx, step in enumerate(self._steps, 1):
                w.writerow([f'[{idx}] {step["name"]}'])
                if step['description']:
                    w.writerow([f'# {step["description"]}'])
                self._write_csv_data(w, step['data'], fmt)
                w.writerow([])

        print(f'[StepLogger] CSV saved → {path}')

    def _write_csv_data(self, writer, data, fmt) -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                writer.writerow([f'## {k}'])
                self._write_csv_data(writer, v, fmt)
            return

        if not isinstance(data, np.ndarray):
            writer.writerow([str(data)])
            return

        arr = np.array(data, dtype=float)
        rows, cols = self._resolve_labels(arr)

        if arr.ndim == 1:
            writer.writerow(rows or [f'A{i+1}' for i in range(arr.shape[0])])
            writer.writerow([fmt.format(v) for v in arr])

        elif arr.ndim == 2:
            col_labels = cols or [f'C{j+1}' for j in range(arr.shape[1])]
            row_labels = rows or [f'A{i+1}' for i in range(arr.shape[0])]
            writer.writerow([''] + col_labels)
            for i, rl in enumerate(row_labels):
                writer.writerow([rl] + [fmt.format(v) for v in arr[i]])

        elif arr.ndim == 3:
            m, n, _ = arr.shape
            row_labels = rows or [f'A{i+1}' for i in range(m)]
            col_labels = cols or [f'C{j+1}' for j in range(n)]
            header = ['']
            for c in col_labels:
                header += [f'{c}_l', f'{c}_m', f'{c}_u']
            writer.writerow(header)
            for i, rl in enumerate(row_labels):
                row = [rl]
                for j in range(n):
                    row += [fmt.format(v) for v in arr[i, j]]
                writer.writerow(row)

    #  Excel                                                               
    def _write_excel(self) -> None:
        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        except ImportError as e:
            raise ImportError(
                'Excel output requires openpyxl: pip install pyfdm[excel]'
            ) from e

        path = self.path.with_suffix('.xlsx')
        path.parent.mkdir(parents=True, exist_ok=True)

        BLUE  = 'FF1F4E79'
        LBLUE = 'FFD6E4F0'
        GRAY  = 'FFF2F2F2'
        WHITE = 'FFFFFFFF'
        GREEN = 'FFE2EFDA'

        thin = Side(style='thin', color='FFCCCCCC')
        def bdr():
            return Border(left=thin, right=thin, top=thin, bottom=thin)

        def hfill(): return PatternFill('solid', fgColor=BLUE)
        def sfill(): return PatternFill('solid', fgColor=LBLUE)
        def hfont(): return Font(bold=True, color='FFFFFFFF', name='Arial', size=10)
        def nfont(): return Font(name='Arial', size=10)
        def bfont(): return Font(bold=True, name='Arial', size=10)

        fmt = f'{{:{self.float_fmt}}}'

        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        # ── Overview sheet ────────────────────────────────────────────────
        ws_ov = wb.create_sheet('Overview', 0)
        ws_ov.column_dimensions['A'].width = 6
        ws_ov.column_dimensions['B'].width = 30
        ws_ov.column_dimensions['C'].width = 50

        # title
        ws_ov.merge_cells('A1:C1')
        c = ws_ov['A1']
        c.value = f'{self._method_name} — Intermediate Computation Steps'
        c.fill = hfill(); c.font = hfont()
        c.alignment = Alignment(horizontal='center')

        ws_ov['A2'] = '#'
        ws_ov['B2'] = 'Step name'
        ws_ov['C2'] = 'Description'
        for col in 'ABC':
            ws_ov[f'{col}2'].fill = sfill()
            ws_ov[f'{col}2'].font = bfont()
            ws_ov[f'{col}2'].border = bdr()

        for idx, step in enumerate(self._steps, 1):
            ws_ov[f'A{idx+2}'] = idx
            ws_ov[f'B{idx+2}'] = step['name']
            ws_ov[f'C{idx+2}'] = step['description']
            fill = PatternFill('solid', fgColor=GRAY if idx % 2 == 0 else WHITE)
            for col in 'ABC':
                cell = ws_ov[f'{col}{idx+2}']
                cell.fill = fill
                cell.font = nfont()
                cell.border = bdr()

        # ── One sheet per step ────────────────────────────────────────────
        for idx, step in enumerate(self._steps, 1):
            sheet_name = f'{idx}. {step["name"]}'[:31]
            ws = wb.create_sheet(title=sheet_name)
            self._write_excel_step(ws, step, idx,
                                   hfill, sfill, hfont, nfont, bfont, bdr,
                                   GRAY, WHITE, fmt, openpyxl)

        wb.save(path)
        print(f'[StepLogger] Excel saved → {path}')

    def _write_excel_step(self, ws, step, idx,
                          hfill, sfill, hfont, nfont, bfont, bdr,
                          GRAY, WHITE, fmt, openpyxl) -> None:
        from openpyxl.styles import Alignment, PatternFill

        data = step['data']
        if not isinstance(data, np.ndarray):
            if isinstance(data, dict):
                row = 1
                for k, v in data.items():
                    ws.cell(row=row, column=1, value=str(k)).font = bfont()
                    if isinstance(v, np.ndarray):
                        row = self._excel_write_array(
                            ws, v, row + 1, fmt, hfill, sfill, hfont,
                            nfont, bfont, bdr, GRAY, WHITE, openpyxl)
                    else:
                        ws.cell(row=row + 1, column=1, value=str(v)).font = nfont()
                        row += 2
            else:
                ws.cell(row=1, column=1, value=str(data)).font = nfont()
            return

        self._excel_write_array(ws, np.array(data, dtype=float), 1, fmt,
                                hfill, sfill, hfont, nfont, bfont, bdr,
                                GRAY, WHITE, openpyxl)

    def _excel_write_array(self, ws, arr, start_row, fmt,
                           hfill, sfill, hfont, nfont, bfont, bdr,
                           GRAY, WHITE, openpyxl) -> int:
        from openpyxl.styles import Alignment, PatternFill

        rows, cols = self._resolve_labels(arr)

        if arr.ndim == 1:
            row_labels = rows or [f'A{i+1}' for i in range(arr.shape[0])]
            for j, rl in enumerate(row_labels):
                c = ws.cell(row=start_row, column=j + 2, value=rl)
                c.fill = sfill(); c.font = bfont(); c.border = bdr()
                c.alignment = Alignment(horizontal='center')
            for j, v in enumerate(arr):
                c = ws.cell(row=start_row + 1, column=j + 2, value=float(v))
                c.font = nfont(); c.border = bdr()
                c.number_format = '0.0000'
                c.alignment = Alignment(horizontal='center')
            ws.column_dimensions[openpyxl.utils.get_column_letter(1)].width = 4
            return start_row + 3

        if arr.ndim == 2:
            m, n = arr.shape
            row_labels = rows or [f'A{i+1}' for i in range(m)]
            col_labels = cols or [f'C{j+1}' for j in range(n)]
            ws.cell(row=start_row, column=1, value='').border = bdr()
            for j, cl in enumerate(col_labels):
                c = ws.cell(row=start_row, column=j + 2, value=cl)
                c.fill = sfill(); c.font = bfont(); c.border = bdr()
                c.alignment = Alignment(horizontal='center')
            for i, rl in enumerate(row_labels):
                fill = PatternFill('solid', fgColor=GRAY if i % 2 == 1 else WHITE)
                c = ws.cell(row=start_row + i + 1, column=1, value=rl)
                c.fill = fill; c.font = bfont(); c.border = bdr()
                for j, v in enumerate(arr[i]):
                    c = ws.cell(row=start_row + i + 1, column=j + 2, value=float(v))
                    c.fill = fill; c.font = nfont(); c.border = bdr()
                    c.number_format = '0.0000'
                    c.alignment = Alignment(horizontal='center')
            ws.column_dimensions[openpyxl.utils.get_column_letter(1)].width = 14
            for j in range(n):
                ws.column_dimensions[openpyxl.utils.get_column_letter(j+2)].width = 12
            return start_row + m + 2

        if arr.ndim == 3:
            m, n, _ = arr.shape
            row_labels = rows or [f'A{i+1}' for i in range(m)]
            col_labels = cols or [f'C{j+1}' for j in range(n)]
            # merged criterion headers
            ws.cell(row=start_row, column=1, value='').border = bdr()
            for j, cl in enumerate(col_labels):
                col_start = 2 + j * 3
                ws.merge_cells(start_row=start_row, start_column=col_start,
                                end_row=start_row, end_column=col_start + 2)
                c = ws.cell(row=start_row, column=col_start, value=cl)
                c.fill = sfill(); c.font = bfont(); c.border = bdr()
                c.alignment = Alignment(horizontal='center')
                for k, sub in enumerate(['l', 'm', 'u']):
                    c2 = ws.cell(row=start_row + 1, column=col_start + k, value=sub)
                    c2.fill = PatternFill('solid', fgColor='FFBDD7EE')
                    c2.font = bfont(); c2.border = bdr()
                    c2.alignment = Alignment(horizontal='center')
            ws.cell(row=start_row + 1, column=1, value='').border = bdr()
            for i, rl in enumerate(row_labels):
                fill = PatternFill('solid', fgColor=GRAY if i % 2 == 1 else WHITE)
                c = ws.cell(row=start_row + i + 2, column=1, value=rl)
                c.fill = fill; c.font = bfont(); c.border = bdr()
                for j in range(n):
                    for k in range(3):
                        c2 = ws.cell(row=start_row + i + 2,
                                     column=2 + j * 3 + k,
                                     value=float(arr[i, j, k]))
                        c2.fill = fill; c2.font = nfont(); c2.border = bdr()
                        c2.number_format = '0.0000'
                        c2.alignment = Alignment(horizontal='center')
            ws.column_dimensions[openpyxl.utils.get_column_letter(1)].width = 14
            for j in range(n):
                for k in range(3):
                    col_letter = openpyxl.utils.get_column_letter(2 + j * 3 + k)
                    ws.column_dimensions[col_letter].width = 8
            return start_row + m + 3

        return start_row + 2

    #  Validation                                                        #
    @staticmethod
    def _parse_output(output) -> set:
        if isinstance(output, str):
            output = [output]
        result = set()
        for fmt in output:
            fmt = fmt.strip().lower()
            if fmt not in _VALID_FORMATS:
                raise ValueError(
                    f"Invalid output format '{fmt}'. "
                    f"Valid options: {_VALID_FORMATS}."
                )
            result.add(fmt)
        return result
