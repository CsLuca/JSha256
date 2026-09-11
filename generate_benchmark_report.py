#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from html import escape


def fmt_float(v, digits=2):
    try:
        return f"{float(v):,.{digits}f}"
    except Exception:
        return "n/a"


def normalize_row(r):
    row = dict(r)
    row.setdefault("engine", "CPU")
    simd = str(row.get("simd", ""))
    if "gpu_api" not in row or not row.get("gpu_api"):
        if row["engine"] == "GPU" and simd.upper() in ("CUDA", "OPENCL"):
            row["gpu_api"] = simd.upper()
        else:
            row["gpu_api"] = "-"
    return row


def latest_json_file(benchmark_dir: Path):
    files = sorted(benchmark_dir.glob("results-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def render_html(data, rows, source_name):
    rows = [normalize_row(r) for r in rows]
    rows_sorted = sorted(rows, key=lambda x: float(x.get("hashes_per_sec", 0.0)), reverse=True)
    top = rows_sorted[:15]
    max_h = max((float(r.get("hashes_per_sec", 0.0)) for r in top), default=1.0)

    cpu_rows = [r for r in rows if r.get("engine") == "CPU"]
    gpu_rows = [r for r in rows if r.get("engine") == "GPU"]

    def count_api(api):
        return len([r for r in gpu_rows if str(r.get("gpu_api", "-")).upper() == api])

    def top_row(rows_local):
        if not rows_local:
            return None
        return max(rows_local, key=lambda x: float(x.get("hashes_per_sec", 0.0)))

    top_cpu = top_row(cpu_rows)
    top_gpu = top_row(gpu_rows)

    bars = []
    for r in top:
        pct = (float(r.get("hashes_per_sec", 0.0)) / max_h) * 100.0 if max_h > 0 else 0
        label = f"{r.get('engine','?')} {r.get('gpu_api','-')} {r.get('simd','?')} {r.get('version','?')}"
        bars.append(
            f"""
            <div class=\"bar-row\">
              <div class=\"bar-label\">{escape(label)}</div>
              <div class=\"bar-track\"><div class=\"bar-fill\" style=\"width:{pct:.2f}%\"></div></div>
              <div class=\"bar-value\">{escape(fmt_float(r.get('hashes_per_sec',0.0),2))} H/s</div>
            </div>
            """
        )

    table_rows = []
    for r in rows_sorted:
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(r.get('engine','')))}</td>"
            f"<td>{escape(str(r.get('gpu_api','')))}</td>"
            f"<td>{escape(str(r.get('simd','')))}</td>"
            f"<td>{escape(str(r.get('version','')))}</td>"
            f"<td>{escape(str(r.get('backend','')))}</td>"
            f"<td class='num'>{escape(str(r.get('lanes','')))}</td>"
            f"<td class='num'>{escape(fmt_float(r.get('hashes_per_sec',0.0),2))}</td>"
            f"<td class='num'>{escape(fmt_float(r.get('std_hashes_per_sec',0.0),2))}</td>"
            f"<td class='num'>{escape(fmt_float(r.get('cycles_per_hash',0.0),2))}</td>"
            f"<td class='num'>{escape(fmt_float(r.get('std_cycles_per_hash',0.0),2))}</td>"
            "</tr>"
        )

    validation_ok = bool(data.get("validation_ok", False))
    validation_text = str(data.get("validation_text", "")).replace("\r\n", "\n")

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>JSha256 Benchmark Report</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --ink: #11203a;
      --muted: #5b6780;
      --line: #d9e0ec;
      --brand: #0f5cc0;
      --brand2: #1fa2ff;
      --good: #1d8f4d;
      --bad: #c62828;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Segoe UI, Tahoma, Arial, sans-serif; color: var(--ink); background: linear-gradient(180deg, #eef3fb 0%, var(--bg) 100%); }}
    .wrap {{ max-width: 1280px; margin: 24px auto; padding: 0 16px; }}
    .header {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; box-shadow: 0 4px 18px rgba(17,32,58,.06); }}
    h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
    .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .v {{ font-size: 21px; font-weight: 700; margin-top: 4px; }}
    .section {{ margin-top: 16px; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
    .section h2 {{ margin: 0 0 12px 0; font-size: 20px; }}
    .bar-row {{ display: grid; grid-template-columns: 330px 1fr 170px; gap: 10px; align-items: center; margin-bottom: 8px; }}
    .bar-label {{ font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .bar-track {{ width: 100%; height: 16px; background: #edf2fa; border-radius: 99px; overflow: hidden; border: 1px solid #e1e8f5; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--brand), var(--brand2)); }}
    .bar-value {{ font-size: 13px; text-align: right; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; position: sticky; top: 0; background: #fbfdff; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .scroll {{ max-height: 480px; overflow: auto; border: 1px solid var(--line); border-radius: 10px; }}
    .ok {{ color: var(--good); font-weight: 700; }}
    .ko {{ color: var(--bad); font-weight: 700; }}
    pre {{ margin: 0; white-space: pre-wrap; color: #24334d; background: #f8fbff; border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
    @media (max-width: 1040px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} .bar-row {{ grid-template-columns: 1fr; }} .bar-value {{ text-align: left; }} }}
    @media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"header\">
      <h1>JSha256 Benchmark Report</h1>
      <div class=\"meta\">Source: <strong>{escape(source_name)}</strong> · Timestamp: <strong>{escape(str(data.get('timestamp','n/a')))}</strong></div>
      <div class=\"grid\">
        <div class=\"card\"><div class=\"k\">Validation</div><div class=\"v {'ok' if validation_ok else 'ko'}\">{'PASS' if validation_ok else 'FAIL'}</div></div>
        <div class=\"card\"><div class=\"k\">Rows</div><div class=\"v\">{len(rows)}</div></div>
        <div class=\"card\"><div class=\"k\">GPU CUDA rows</div><div class=\"v\">{count_api('CUDA')}</div></div>
        <div class=\"card\"><div class=\"k\">GPU OpenCL rows</div><div class=\"v\">{count_api('OPENCL')}</div></div>
      </div>
      <div class=\"grid\" style=\"margin-top:10px\">
        <div class=\"card\"><div class=\"k\">Top CPU</div><div class=\"v\">{escape(fmt_float(top_cpu.get('hashes_per_sec',0.0),2) if top_cpu else 'n/a')} H/s</div><div class=\"meta\">{escape((top_cpu.get('simd','') + ' ' + top_cpu.get('version','')) if top_cpu else 'n/a')}</div></div>
        <div class=\"card\"><div class=\"k\">Top GPU</div><div class=\"v\">{escape(fmt_float(top_gpu.get('hashes_per_sec',0.0),2) if top_gpu else 'n/a')} H/s</div><div class=\"meta\">{escape((top_gpu.get('gpu_api','') + ' ' + top_gpu.get('version','')) if top_gpu else 'n/a')}</div></div>
      </div>
    </div>

    <div class=\"section\">
      <h2>Top 15 Throughput (Hash/s)</h2>
      {''.join(bars)}
    </div>

    <div class=\"section\">
      <h2>Validation Detail</h2>
      <pre>{escape(validation_text)}</pre>
    </div>

    <div class=\"section\">
      <h2>All Rows</h2>
      <div class=\"scroll\">
        <table>
          <thead>
            <tr>
              <th>Engine</th>
              <th>API</th>
              <th>SIMD</th>
              <th>Ver</th>
              <th>Backend</th>
              <th>Lanes</th>
              <th>Hash/s</th>
              <th>Std H/s</th>
              <th>Cycles/hash</th>
              <th>Std Cyc</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate a professional HTML benchmark report from JSha256 JSON export")
    parser.add_argument("--input", "-i", help="Path to results-*.json")
    parser.add_argument("--output", "-o", help="Output HTML file path")
    parser.add_argument("--open", action="store_true", help="Open generated report in browser")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    benchmark_dir = repo_root / "benchmark-output"

    if args.input:
        in_path = Path(args.input)
    else:
        in_path = latest_json_file(benchmark_dir)
        if in_path is None:
            raise SystemExit("No JSON benchmark export found in benchmark-output/. Run the benchmark first.")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])

    out_path = Path(args.output) if args.output else benchmark_dir / f"report-{data.get('timestamp', 'latest')}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(data, rows, in_path.name)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written: {out_path}")

    if args.open:
        import webbrowser
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
