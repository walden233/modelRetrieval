"""CSV and offline HTML from independent semantic experiment results."""

import csv
import html
import json
import os
from pathlib import Path

from bise.common.paths import resolve_path
from bise.modalities.semantic.store import read_jsonl


FIELDS = ['suite', 'cfg', 'variant', 'provider', 'model', 'frames', 'direction',
          'level', 'representation', 'metric', 'value', 'partial']


def export_report(directory: Path) -> Path:
    rows, sections = [], []
    esc = lambda value: html.escape(str(value))
    params = json.loads((directory / 'params.json').read_text())
    allowed = [rep for rep in params.get('comparison_representations', params['config']['representations'])
               if rep not in params.get('disabled', [])]
    active = [rep for rep in allowed
              if (directory / rep / 'annotations.jsonl').exists()]

    def add_metrics(payload, suite, variant, provider, model, frames):
        for direction, levels in payload['metrics'].items():
            for level, representations in levels.items():
                for rep, metrics in representations.items():
                    if rep not in (allowed if suite == 'cost' else active):
                        continue
                    for metric, value in metrics.items():
                        rows.append(dict(suite=suite, cfg=payload['cfg'], variant=variant,
                                         provider=provider, model=model, frames=frames,
                                         direction=direction, level=level, representation=rep,
                                         metric=metric, value=value, partial=payload['partial']))

    def add_engineering(data, suite, variant, provider, model, frames, level, partial):
        for metric, value in data.items():
            if isinstance(value, (int, float)) or value is None:
                rows.append(dict(suite=suite, cfg='all', variant=variant, provider=provider,
                                 model=model, frames=frames, direction='', level=level,
                                 representation='', metric=metric, value=value, partial=partial))

    engineering = params.get('annotation_engineering', {})
    for kind, data in engineering.items():
        signature = json.loads((resolve_path(data['job']) / 'params.json').read_text())['signature']
        add_engineering(data['engineering'], 'main', Path(data['job']).name,
                        signature['provider'], signature['model'], signature.get('frames', 0),
                        'annotation', data['engineering'].get('coverage_rate', 0) < 1)
        add_engineering(data['last_execution'], 'main', Path(data['job']).name,
                        signature['provider'], signature['model'], signature.get('frames', 0),
                        'last_execution', data['last_execution']['selected'] < sum(
                            count for splits in params['counts'].values() for count in splits.values()))
    if engineering:
        sections.append('<h2>Annotation timing and cost</h2><pre>' +
                        esc(json.dumps(engineering, indent=2)) + '</pre>')
    for rep, data in params.get('encoding_engineering', {}).items():
        if rep in active:
            add_engineering(data, 'main', rep, '', '', '', 'encoding', False)
    for path in sorted((directory / 'evaluations').glob('*/metrics.json')):
        payload = json.loads(path.read_text())
        for rep in active:
            if rep not in payload['metrics'].get('human_to_robot', {}).get('task', {}):
                continue
            source = directory / rep / 'evaluations' / payload['cfg'] / 'metrics.json'
            if not source.exists():
                continue
            settings = json.loads((directory / rep / 'params.json').read_text())
            signature = settings['signature']
            add_metrics(json.loads(source.read_text()), 'main', Path(settings['job']).name,
                        signature['provider'], signature['model'], signature['frames'])
        sections.append(f"<h2>{esc(payload['cfg'])} / {esc(payload['split'])}</h2>"
                        f"<p>Coverage: {payload['coverage']['present']}/{payload['coverage']['expected']}; "
                        f"partial={payload['partial']}. <a href='{path.relative_to(directory)}'>Metrics</a></p>"
                        f"<img src='charts/{esc(payload['cfg'])}.png' alt='retrieval comparison'>")
    cost_path = directory / 'cost/summary.json'
    if cost_path.exists():
        cost = json.loads(cost_path.read_text())
        sections.append(f"<h2>Cost subset</h2><p>Coverage: {cost['common_videos']}/{cost['target_videos']}. "
                        "One scene per task: task and scene positives coincide. "
                        "<a href='cost/summary.json'>Raw metrics and engineering data</a></p>")
        if (directory / 'charts/cost.png').is_file():
            sections.append("<img src='charts/cost.png' alt='cost and latency'>")
        for variant in cost['variants']:
            add_metrics({'cfg': cost['cfg'], 'metrics': variant['metrics'],
                         'partial': variant['partial']}, 'cost', variant['name'],
                        variant['provider'], variant['model'], variant['frames'])
            add_engineering(variant['engineering'], 'cost', variant['name'], variant['provider'],
                            variant['model'], variant['frames'], 'annotation', variant['partial'])
    analysis_path = directory / 'analysis.json'
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        if analysis['views']:
            sections.append('<h2>Independent-video view consistency</h2>'
                            "<img src='charts/views.png' alt='cross-view discrimination'>")
        sections.append('<h2>Label diagnostics</h2><pre>' +
                        esc(json.dumps(analysis['labels'], indent=2)) + '</pre>')
        if 'multimodal' in analysis:
            sections.append('<h2>Complementarity on common query/gallery subsets</h2><pre>' +
                            esc(json.dumps(analysis['multimodal'], indent=2)) + '</pre>')
        rank_rows = [row for path in (directory / 'evaluations').glob('*/rankings.jsonl')
                     for row in read_jsonl(path)]
        sections.append('<h2>Same-video representation examples</h2>')
        for name, example in analysis['examples'].items():
            video = os.path.relpath(resolve_path(example['video_path']), directory)
            sections.append(f"<h3>{esc(example['sample_id'])}</h3>"
                            f"<video controls preload='none' src='{esc(video)}'></video>"
                            f"<p><a href='charts/cases/{esc(name)}.png'>Frames and task descriptions</a></p>")
            for rep, text in example['texts'].items():
                if rep not in active:
                    continue
                ranks = [row for row in rank_rows if row['query_id'] == example['sample_id'] and
                         row['representation'] == rep and row['level'] == 'scene']
                rank = ranks[0]['best_rank'] if ranks else 'not evaluated'
                sections.append(f'<details><summary>{rep}: scene best-positive rank {rank}</summary>'
                                f'<pre>{esc(text)}</pre></details>')
    with (directory / 'results.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    selected = [row for row in rows if row['suite'] == 'main' and
                row['direction'] == 'human_to_robot' and row['metric'] in ('R@1', 'MRR', 'NDCG@10')]
    table = '<table><tr>' + ''.join(f'<th>{field}</th>' for field in FIELDS) + '</tr>'
    table += ''.join('<tr>' + ''.join(f'<td>{esc(row[field])}</td>' for field in FIELDS) + '</tr>'
                     for row in selected) + '</table>'
    document = "<!doctype html><meta charset='utf-8'><title>Semantic chapter</title>" \
        '<style>body{font:15px sans-serif;max-width:1200px;margin:32px auto;padding:0 20px}' \
        'img{max-width:100%}video{max-width:640px}pre{white-space:pre-wrap;overflow-wrap:anywhere}' \
        'td,th{border:1px solid #ddd;padding:6px}table{border-collapse:collapse}' \
        'details{padding:8px;background:#f5f5f5;margin:6px 0}</style>' \
        "<h1>Semantic retrieval</h1><p><a href='results.csv'>All comparison data (CSV)</a> · " \
        "<a href='params.json'>Settings and timing</a> · <a href='reviews.csv'>Review worksheet</a></p>" \
        + table + '\n'.join(sections)
    path = directory / 'report.html'
    path.write_text(document, encoding='utf-8')
    return path
