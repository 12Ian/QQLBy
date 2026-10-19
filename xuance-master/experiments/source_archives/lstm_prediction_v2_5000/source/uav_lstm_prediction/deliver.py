"""Create the review gallery and checksummed delivery package on the server."""
import json
from pathlib import Path
import zipfile
from .core import sha256,write_json

def deliver(run):
    run=Path(run)
    p=json.loads((run/"protocol.json").read_text())
    summary=json.loads((run/"training_summary.json").read_text())
    dataset=json.loads((run/"dataset_summary.json").read_text())
    comparisons=json.loads((run/"data_expansion_comparison.json").read_text())
    collection=json.loads((run/"collection.json").read_text())
    test=json.loads((run/"test_metrics.json").read_text())
    videos=json.loads((run/"videos/video_manifest.json").read_text())
    overview=dict(total_episodes=len(collection["episodes"]),total_scenes=collection["paired_scenes"],
                  total_decision_steps=sum(e["steps"] for e in collection["episodes"]),
                  total_simulation_seconds=sum(e["seconds"] for e in collection["episodes"]),
                  splits={split:dict(episodes=sum(e["split"]==split for e in p["episodes"]),
                       scenes=len({(e["level"],e["seed"]) for e in p["episodes"] if e["split"]==split}))
                       for split in ["train","validation","test"]})
    write_json(run/"dataset_overview.json",overview)
    def row(name,m):
        a=m["all"];h={x["seconds"]:x["mean_error_m"] for x in a["horizons"]}
        return f"| {name} | {a['ade_m']:.3f} | {h[1.]:.3f} | {h[3.]:.3f} | {h[6.]:.3f} | {a['p90_m']:.3f} | {a['p95_m']:.3f} |"
    lines=["# LSTM 数据扩充实验：500 → 5,000 回合","",
           "全部采集、训练、评估和视频渲染在服务器完成；尚未接入 MATD3。","",
           "## 数据与训练","",
           "- 2,500 个独立初始场景，每级 500 个；两种冻结策略分别执行，共 5,000 回合。",
           "- 复用原有 500 回合，新增 4,500 回合；原场景的训练/验证/测试归属保持不变。",
           "- 每级 350/70/80 个训练/验证/测试场景，总计 1,750/350/400 个场景。",
           f"- 训练窗口 {dataset['train']['windows']:,}；验证窗口 {dataset['validation']['windows']:,}；测试窗口 {test['samples']:,}，均包含四机视角及重叠窗口。",
           "- 仍使用五套固定城市布局及两种固定策略；此实验检验新初态泛化，未检验换城市或新策略。",
           "- 输入、两层 128 单元 LSTM、损失、学习率、batch 256 均保持原设置。Float32 窗口经逐元素一致性验证；采用内存映射存储；训练缓存仅在显存预留足够时加载 GPU，验证数据分批读取。",
           f"- 共训练 {summary['epochs']} 轮，最佳第 {summary['best_epoch']} 轮；验证 1–3 秒误差 {summary['best_validation_error_m']:.3f} 米。",
           "- 按验证集选择模型；原测试集仅用于前后对照，新增测试集未参与训练或选模。",""]
    for key,title in [("test_reference","原测试场景：相同轨迹前后对照"),("test_new","新增独立测试场景")]:
        report=comparisons[key]
        lines+=["## "+title,"",f"{report['scenes']} 场景、{report['episodes']} 回合、{report['samples']:,} 窗口。误差单位为米。","",
                "| 方法 | ADE | 1 秒 | 3 秒 | 6 秒 | P90 | P95 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for name,metrics in report["methods"].items():lines.append(row(name,metrics))
        lines.append("")
    lines+=["## 视频与复现","",
            "index.html 提供 1080p 总览及十个原速案例。沿用训练前固定的原十个测试场景，便于观察扩充前后的差异。",
            "紫色为 LSTM 预测，灰色为匀速预测，绿色为事后真实未来。未来仅供画图和评分；终止后不补标签。",
            "视频使用同一 P1 合法观察视角；完整建筑地图仅供绘图。",
            "data_expansion_comparison.json 提供原测试集与新增测试集的分级、事件、逐时距和尾部误差。",
            "测试窗口重叠且四机视角相关，不能视为独立轨迹；离线预测改善不代表追捕成功率已经改善。","",
            "服务器实验目录："+str(run)]
    lines += ["", "复现：在服务器设置 `PYTHONPATH="+str(run/"source")+"`，再运行 `python -m uav_lstm_prediction.pipeline --run "+str(run)+"`。原始轨迹保存在该目录的 data/ 中。"]
    (run/"REPORT.md").write_text("\n".join(lines)+"\n")
    cards=[]
    for case in videos["cases"]:
        folder=str(Path(case["original"]).parent)
        cards.append(f'<article><h3>等级 {case["level"]+1} · seed {case["seed"]}</h3><video controls preload="none" poster="{folder}/preview.png" src="{case["original"]}"></video><p><a href="{case["original"]}">原速 MP4</a> · <a href="{case["fast"]}">10 倍速</a></p></article>')
    tables=[]
    for key,title in [("test_reference","原测试场景：与旧模型对照"),("test_new","新增独立测试场景")]:
        rows=[]
        for name,metrics in comparisons[key]["methods"].items():
            a=metrics["all"];h={x["seconds"]:x["mean_error_m"] for x in a["horizons"]}
            rows.append(f'<tr><td>{name}</td><td>{a["ade_m"]:.3f}</td><td>{h[1.]:.3f}</td><td>{h[3.]:.3f}</td><td>{h[6.]:.3f}</td></tr>')
        tables.append('<h2>'+title+'</h2><table><tr><th>方法</th><th>ADE / m</th><th>1s / m</th><th>3s / m</th><th>6s / m</th></tr>'+''.join(rows)+'</table>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LSTM 5,000 回合预测评估</title><style>
body{margin:0;background:#f1f3f7;color:#202538;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:32px}h1{font-size:34px;margin-bottom:8px}h2{margin-top:32px}p{color:#555d72}video,img{display:block;width:100%;border-radius:10px;background:#fff}article{background:white;border-radius:14px;padding:18px;box-shadow:0 2px 12px #1724420a}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}a{color:#6540b4}table{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden}td,th{padding:12px 18px;text-align:left;border-bottom:1px solid #ececf1}th{background:#e5e0f4}.note{padding:18px;background:#e9e4f7;border-radius:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:18px}}
</style><main><h1>LSTM 轨迹预测 · 5,000 回合</h1>'''
    html+=f'<p>2,500 个独立场景 · 训练窗口 {dataset["train"]["windows"]:,} · 最佳第 {summary["best_epoch"]} 轮</p>'
    html+='<p class="note">本轮仅扩充预测器数据并重新训练。MATD3 尚未接入；请先查看预测轨迹。视频保留原十个固定测试案例，便于与上一版比较。</p>'
    html+='<h2>10 倍速总览</h2><video controls preload="metadata" src="videos/all_levels_prediction_10x.mp4"></video>'
    html+=''.join(tables)+'<h2>逐时距误差对照</h2><img src="data_expansion_comparison.png" alt="原测试集与新增测试集误差">'
    html+='<h2>训练曲线</h2><img src="learning_curve.png" alt="训练与验证曲线"><h2>十个固定案例 · 原速</h2><div class="grid">'+''.join(cards)+'</div>'
    html+='<h2>报告与来源</h2><p><a href="REPORT.md">完整报告</a> · <a href="data_expansion_comparison.json">详细误差</a> · <a href="protocol.json">数据划分与来源</a> · <a href="model/best.pt">模型检查点</a></p><p>紫色：LSTM · 灰色：匀速 · 绿色：真实未来，仅供事后展示。所有误差单位为米。</p></main></html>'
    (run/"index.html").write_text(html)
    files=[x for x in run.rglob("*") if x.is_file() and (
        x.parent==run and x.suffix in [".json",".png",".md",".html",".csv"] or
        "videos" in x.relative_to(run).parts and x.suffix in [".mp4",".png",".jpg",".json"] or
        x==run/"model/best.pt" or "source" in x.relative_to(run).parts and x.suffix==".py")]
    files=[x for x in files if x.name not in ["delivery_manifest.json"]]
    write_json(run/"delivery_manifest.json",dict(files=[dict(path=str(x.relative_to(run)),bytes=x.stat().st_size,sha256=sha256(x)) for x in sorted(files)]))
    with zipfile.ZipFile(run/"delivery_bundle.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for x in files+[run/"delivery_manifest.json"]:archive.write(x,x.relative_to(run))
    print("DELIVERY_BUNDLE",str(run/"delivery_bundle.zip"),(run/"delivery_bundle.zip").stat().st_size,flush=True)
