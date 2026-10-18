"""Readable cost comparisons from saved metrics only; no model or API access."""

import csv
import html
import json
from pathlib import Path

import numpy as np

from bise.modalities.semantic.representations import REPRESENTATIONS
from bise.modalities.semantic.store import write_json


METRICS = ('R@1', 'R@5', 'R@10', 'MRR', 'NDCG@10')


def variant_label(row: dict) -> str:
    return f"{row['provider'].upper()} / {row['frames']}f"


def draw_cost_overview(data: dict, target: Path, dpi: int = 150) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = data['variants']
    if not rows:
        raise ValueError('No cost variants to plot')
    reps = [rep for rep in REPRESENTATIONS if any(
        rep in row['metrics'].get('human_to_robot', {}).get('task', {}) for row in rows)]
    rep = 'S1' if 'S1' in reps else reps[0]
    labels = [variant_label(row) for row in rows]
    colors = plt.get_cmap('tab10')(np.arange(len(rows)))
    figure, axes = plt.subplots(2, 2, figsize=(17, 10.5), constrained_layout=True)
    axis = axes[0, 0]
    width = .8 / len(rows)
    for index, row in enumerate(rows):
        metrics = row['metrics'].get('human_to_robot', {}).get('task', {}).get(rep, {})
        values = [metrics.get(metric, np.nan) * 100 for metric in METRICS]
        bars = axis.bar(np.arange(len(METRICS)) + (index - (len(rows) - 1) / 2) * width,
                        values, width, color=colors[index], label=labels[index])
        axis.bar_label(bars, fmt='%.1f', fontsize=7, padding=2)
    axis.set(xticks=np.arange(len(METRICS)), xticklabels=METRICS, ylim=(0, 108),
             ylabel='Score (%)', title=f'{rep} retrieval / human to robot / higher is better')
    axis.legend(fontsize=8, ncol=3, loc='upper left')
    axis.grid(axis='y', alpha=.15)

    axis = axes[0, 1]
    matrix = np.array([[row['metrics'].get('human_to_robot', {}).get('task', {})
                        .get(group, {}).get('MRR', np.nan) * 100 for group in reps]
                       for row in rows])
    image = axis.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
    for y, x in np.ndindex(matrix.shape):
        value = matrix[y, x]
        axis.text(x, y, f'{value:.1f}' if np.isfinite(value) else 'N/A',
                  ha='center', va='center', color='white' if value >= 65 else '#162433')
    axis.set(xticks=np.arange(len(reps)), xticklabels=reps,
             yticks=np.arange(len(rows)), yticklabels=labels,
             title='All representations / human to robot MRR (%)')
    figure.colorbar(image, ax=axis, fraction=.04, pad=.02)

    axis = axes[1, 0]
    costs = [row['engineering'].get('known_cost', np.nan) for row in rows]
    bars = axis.bar(np.arange(len(rows)), costs, color=colors)
    for bar, row, cost in zip(bars, rows, costs):
        missing = row['engineering'].get('unknown_cost_attempts', 0)
        unit = row['engineering'].get('known_cost_per_video')
        label = f'{cost:.3f} CNY\n{unit:.4f}/video' if unit is not None else f'{cost:.3f} CNY'
        if missing:
            label += f'\n{missing} unpriced attempts'
        axis.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontsize=8)
    axis.set(xticks=np.arange(len(rows)), xticklabels=labels, ylabel='Known API cost (CNY)',
             ylim=(0, max([c for c in costs if np.isfinite(c)] + [1]) * 1.4),
             title='Known annotation cost / entire selected set / lower is better')
    axis.grid(axis='y', alpha=.15)

    axis = axes[1, 1]
    for offset, field, label, color in ((-.2, 'request_seconds_p50', 'P50', '#4178ac'),
                                       (.2, 'request_seconds_p95', 'P95', '#b4cce2')):
        values = [row['engineering'].get(field) for row in rows]
        bars = axis.bar(np.arange(len(rows)) + offset,
                        [v if v is not None else np.nan for v in values], .4,
                        label=label, color=color)
        axis.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    axis.set(xticks=np.arange(len(rows)), xticklabels=labels, ylabel='API response time (seconds)',
             title='Request latency / cache reads and pacing wait excluded')
    axis.legend()
    axis.margins(y=.2)
    axis.grid(axis='y', alpha=.15)
    for axis in (axes[0, 0], axes[1, 0], axes[1, 1]):
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle(
        f"Cost experiment / {data['cfg']} / {data['common_videos']}/{data['target_videos']} videos"
        f" / {data.get('tasks', '?')} tasks / one scene per task\n"
        'Task and scene positives coincide. Costs include recorded retries; unpriced attempts excluded.',
        fontsize=13)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi)
    plt.close(figure)


def export_cost_report(directory: Path) -> list[Path]:
    source = directory / 'cost/summary.json'
    if not source.is_file():
        return []
    data = json.loads(source.read_text())
    output = source.parent
    image = output / 'overview.png'
    index_path = directory / 'plots.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index['cost/overview.png'] = {'kind': 'semantic', 'inputs': ['cost/summary.json']}
    write_json(index_path, index)
    draw_cost_overview(data, image)
    fields = ['variant', 'provider', 'model', 'frames', 'direction', 'representation',
              *METRICS, 'MPR-all', 'known_cost', 'known_cost_per_video', 'unknown_cost_attempts',
              'request_seconds_p50', 'request_seconds_p95', 'measured_items_per_second']
    csv_path = output / 'comparison.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in data['variants']:
            for direction, levels in variant['metrics'].items():
                for rep, metrics in levels.get('task', {}).items():
                    row = dict(variant=variant['name'], provider=variant['provider'],
                               model=variant['model'], frames=variant['frames'],
                               direction=direction, representation=rep)
                    row.update({key: metrics[key] for key in (*METRICS, 'MPR-all') if key in metrics})
                    row.update({key: variant['engineering'].get(key) for key in fields[12:]})
                    writer.writerow(row)
    # Inline data makes the page usable over file:// without a server or network access.
    embedded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    title = html.escape(f"成本实验 · {data['cfg']} · {data['common_videos']}/{data['target_videos']} 段视频")
    first = data['variants'][0]['metrics'].get('human_to_robot', {}).get('task', {})
    metric = next(iter(first.values()), {})
    scope = (f"{data.get('tasks', '?')}个任务，每任务1个scene，每scene两对人机视频；"
             f"主方向{metric.get('valid_queries', '?')}条查询、"
             f"{metric.get('gallery_size', '?')}条候选。")
    document = (_PAGE.replace('__TITLE__', title).replace('__SCOPE__', html.escape(scope))
                .replace('__DATA__', embedded))
    page = output / 'index.html'
    page.write_text(document, encoding='utf-8')
    return [page, image, csv_path]


_PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>语义标注成本实验</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font:15px system-ui,sans-serif;color:#182b42;background:#f2f5fa;margin:0}
main{max-width:1400px;margin:30px auto;padding:0 24px}h1{font-size:26px}h2{font-size:19px}
.card{background:white;border-radius:12px;padding:20px;margin:18px 0;overflow:auto}
.note{color:#56677e;line-height:1.8}a{color:#1763ad}select{padding:6px;margin:0 22px 0 8px}
table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{padding:11px;text-align:right;
border-bottom:1px solid #e5eaf0}th{background:#f5f8fc}td:first-child,th:first-child{text-align:left}
.best{font-weight:750;color:#067044;background:#e3f7ec}.heat td{text-align:center}
img{width:100%;height:auto}.small{font-size:13px}code{background:#eaf0f7;padding:3px 6px}
</style><main><h1>__TITLE__</h1>
<p class="note">__SCOPE__
因此task与scene正样本集合一致。所有方案比较同一批视频。</p>
<p><a href="overview.png">下载总览图片</a> · <a href="comparison.csv">下载比较数据 CSV</a> ·
<a href="summary.json">原始指标与工程记录</a></p>
<div class="card"><label>检索方向<select id="direction"><option value="human_to_robot">人类 → 机器人</option>
<option value="robot_to_human">机器人 → 人类</option></select></label>
<label>表示方式<select id="rep"></select></label>
<h2>检索性能（%，越高越好）</h2><div id="metrics"></div>
<p class="note small">绿色标出当前列最优值；未做显著性检验。主方向为人类→机器人。</p></div>
<div class="card"><h2>六种表示的完整比较</h2><label>指标<select id="metric">
<option>MRR</option><option>R@1</option><option>R@5</option><option>R@10</option><option>NDCG@10</option>
</select></label><div id="heatmap"></div></div>
<div class="card"><h2>费用与效率（一次联合标注生成全部表示）</h2><div id="engineering"></div>
<p class="note small">已知费用包含当前标注记录中的重试；未定价请求的实际费用未知，不按免费计算。
API延迟排除缓存读取与本地限速等待。吞吐受并发、限速、解码和网络影响，不代表模型固有速度；
缺少一次完整运行的测量时显示“—”。GLM 16帧复用主实验记录，不能用本次缓存处理时间比较标注耗时。
费用仅覆盖当前保留的请求记录，已删除记录不包含在内。</p></div>
<div class="card"><h2>固定总览：S1、人类→机器人</h2><img src="overview.png" alt="性能、全表示MRR、已知费用、API延迟总览"></div>
<p class="note small">重绘（不调用API）：<code>python runs/semantic/plot_cost.py</code>；原始依赖：cost/summary.json。</p>
</main><script>
const data=__DATA__, rows=data.variants, keys=['R@1','R@5','R@10','MRR','NDCG@10'];
const reps=[...new Set(rows.flatMap(r=>Object.keys(r.metrics.human_to_robot.task)))];
const el=id=>document.getElementById(id), esc=x=>String(x).replace(/[&<>"']/g,
 c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=2)=>x==null?'—':Number(x).toFixed(n), label=r=>`${r.provider.toUpperCase()} / ${r.frames}帧`;
el('rep').innerHTML=reps.map(r=>`<option>${esc(r)}</option>`).join('');
function table(headers,body,cls=''){return `<table class="${cls}"><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;}
function update(){
 const direction=el('direction').value,rep=el('rep').value,metric=el('metric').value;
 const values=rows.map(r=>r.metrics[direction]?.task?.[rep]||{});
 const max=Object.fromEntries(keys.map(k=>[k,Math.max(...values.map(v=>v[k]??-Infinity))]));
 el('metrics').innerHTML=table(['方案',...keys],rows.map((r,i)=>`<tr><td title="${esc(r.model)}">${esc(label(r))}</td>`+
 keys.map(k=>`<td class="${values[i][k]!=null&&values[i][k]===max[k]?'best':''}">${fmt(values[i][k]==null?null:values[i][k]*100)}</td>`).join('')+'</tr>').join(''));
 el('heatmap').innerHTML=table(['方案',...reps],rows.map(r=>`<tr><td>${esc(label(r))}</td>`+reps.map(p=>{
 const v=r.metrics[direction]?.task?.[p]?.[metric];
 return `<td style="background:${v==null?'white':`hsl(210 65% ${98-v*48}%)`}">${fmt(v==null?null:v*100)}</td>`;
 }).join('')+'</tr>').join(''),'heat');
}
el('engineering').innerHTML=table(['方案','覆盖','已知费用/元','元/视频','未知费用请求','API P50/秒','API P95/秒','完整轮次/秒','吞吐/视频每秒'],rows.map(r=>{
 const e=r.engineering;
 return '<tr>'+[label(r),`${e.successful}/${e.selected}`,fmt(e.known_cost,3),fmt(e.known_cost_per_video,5),
 e.unknown_cost_attempts,fmt(e.request_seconds_p50),fmt(e.request_seconds_p95),
 fmt(e.measured_items_per_second>0?e.selected/e.measured_items_per_second:null),
 fmt(e.measured_items_per_second)].map(v=>`<td>${esc(v)}</td>`).join('')+'</tr>';
}).join(''));
['direction','rep','metric'].forEach(id=>el(id).addEventListener('change',update));update();
</script></html>'''
