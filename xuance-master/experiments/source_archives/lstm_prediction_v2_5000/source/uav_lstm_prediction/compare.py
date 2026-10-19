"""Compare frozen v1 and expanded-data v2 on identical held-out trajectories."""
import json
from pathlib import Path
import numpy as np
import torch
from .train import assemble,evaluate
from .core import ContextLSTM,write_json,sha256,TAUS

def compare(run):
    run=Path(run)
    p=json.loads((run/"protocol.json").read_text())
    parent=Path(p["parent_run"])
    torch.set_num_threads(2)
    device=torch.device("cuda:0")
    models={}
    for label,folder in [("lstm_v1",parent),("lstm_v2",run)]:
        saved=torch.load(folder/"model/best.pt",map_location=device,weights_only=True)
        model=ContextLSTM().to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        models[label]=model
    reference={(e["level"],e["seed"]) for e in json.loads((parent/"protocol.json").read_text())["episodes"] if e["split"]=="test"}
    reports={}
    for split in ["test_reference","test_new"]:
        q=dict(p,episodes=[])
        for e in p["episodes"]:
            if e["split"]!="test":continue
            original=(e["level"],e["seed"]) in reference
            if original!=(split=="test_reference"):continue
            q["episodes"].append(dict(e,split=split))
        data=assemble(run,q,split)
        report=dict(samples=len(data["x"]),episodes=len(data["sources"]),
                    scenes=len({(e["level"],e["seed"]) for e in q["episodes"]}),methods={})
        for label,model in models.items():
            metrics=evaluate(model,data,device)
            report["methods"][label]=metrics["methods"]["lstm"]
            if label=="lstm_v2":
                report["methods"].update({k:v for k,v in metrics["methods"].items() if k!="lstm"})
        reports[split]=report
        write_json(run/(split+"_comparison.json"),report)
        if split=="test_reference":
            old=json.loads((parent/"test_metrics.json").read_text())
            delta=abs(old["methods"]["lstm"]["all"]["ade_m"]-report["methods"]["lstm_v1"]["all"]["ade_m"])
            assert delta<1e-6,delta
            write_json(run/"reference_reproduction.json",dict(passed=True,ade_absolute_difference_m=delta,
                       original_test_episodes_preserved=80,selection_used_test=False))
        del data
    write_json(run/"data_expansion_comparison.json",reports)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),dpi=160)
    colors=dict(lstm_v1="#c05780",lstm_v2="#5d42b3",cv="#888888",ca="#c58b2b")
    for ax,(split,report) in zip(axes,reports.items()):
        for name,metrics in report["methods"].items():
            h=metrics["all"]["horizons"]
            ax.plot([v["seconds"] for v in h],[v["mean_error_m"] for v in h],label=name,color=colors[name])
        ax.set(xlabel="Forecast horizon (s)",ylabel="Mean position error (m)",
               title="Original held-out scenes" if split=="test_reference" else "New held-out scenes")
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(run/"data_expansion_comparison.png");plt.close(fig)
    print("EXPANSION_COMPARISON_COMPLETE",flush=True)
    return reports

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True)
    compare(parser.parse_args().run)
