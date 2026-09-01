from __future__ import annotations

import argparse, csv, json
from pathlib import Path

CHOICES = "abcd"
KS = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40]

def read(path):
    with Path(path).open(encoding="utf-8") as f: return {r["id"]:r for r in map(json.loads,filter(str.strip,f))}
def p(row): return [float(row[f"score_{x}"]) for x in CHOICES]
def lab(v): return CHOICES[max(range(4),key=v.__getitem__)]
def mix(a,b,w=.5): return [(1-w)*x+w*y for x,y in zip(a,b)]
def margin(v):
    s=sorted(v,reverse=True); return s[0]-s[1]
def write(path,ids,answers):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f); w.writerow(["id","answer"]); w.writerows(zip(ids,answers))

def rows(ids, ba, bb, std, h896, h1024, with_truth):
    out=[]
    for i in ids:
        base=[(x+y)/2 for x,y in zip(p(ba[i]),p(bb[i]))]
        anchor=mix(base,p(std[i])); x896=mix(base,p(h896[i])); x1024=mix(base,p(h1024[i]))
        la,l896,l1024=lab(anchor),lab(x896),lab(x1024); alt=[(x+y)/2 for x,y in zip(x896,x1024)]; la2=lab(alt)
        chosen=CHOICES.index(la2)
        r={"id":i,"anchor":la,"alt":la2,"eligible":l896==l1024 and la2!=la,
           "consensus_margin":margin(alt),"min_margin":min(margin(x896),margin(x1024)),
           "support_gain":alt[chosen]-anchor[chosen],"anchor_weak":-margin(anchor)}
        r["balanced"]=r["support_gain"]+.35*r["min_margin"]+.15*r["anchor_weak"]
        if with_truth: r["truth"]=std[i]["true_label"]
        out.append(r)
    return out

def main():
    ap=argparse.ArgumentParser()
    for n in ["base-dev","new-dev","base-test","new-test"]: ap.add_argument("--"+n,nargs=2 if "base" in n else 3,required=True)
    ap.add_argument("--sample",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--prefix",default="qwen35_public_anchor")
    a=ap.parse_args(); bd=list(map(read,a.base_dev)); nd=list(map(read,a.new_dev)); bt=list(map(read,a.base_test)); nt=list(map(read,a.new_test))
    dids=list(bd[0]); tids=[r["id"] for r in csv.DictReader(open(a.sample,encoding="utf-8-sig"))]
    dr=rows(dids,*bd,*nd,True); tr=rows(tids,*bt,*nt,False)
    anchor_acc=sum(r["anchor"]==r["truth"] for r in dr)/len(dr); report={"dev_samples":len(dr),"anchor_accuracy":anchor_acc,"rankings":{},"candidates":[]}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); seen=set()
    metrics=["balanced","support_gain","min_margin","consensus_margin","anchor_weak"]
    for metric in metrics:
        eligible=sorted((r for r in dr if r["eligible"]),key=lambda r:r[metric],reverse=True)
        test_eligible=sorted((r for r in tr if r["eligible"]),key=lambda r:r[metric],reverse=True)
        vals=[]
        for k in KS:
            ds={r["id"] for r in eligible[:k]}; ts={r["id"] for r in test_eligible[:k]}
            acc=sum((r["alt"] if r["id"] in ds else r["anchor"])==r["truth"] for r in dr)/len(dr)
            rescue=sum(r["id"] in ds and r["anchor"]!=r["truth"] and r["alt"]==r["truth"] for r in dr)
            harm=sum(r["id"] in ds and r["anchor"]==r["truth"] and r["alt"]!=r["truth"] for r in dr)
            vals.append({"k":k,"accuracy":acc,"rescue":rescue,"harm":harm})
            answers=tuple(r["alt"] if r["id"] in ts else r["anchor"] for r in tr)
            if answers in seen: continue
            seen.add(answers); name=f"submission_{a.prefix}_{metric}_top{k}.csv"; write(out/name,tids,answers)
            report["candidates"].append({"file":name,"metric":metric,"k":k,"dev_accuracy":acc,"rescue":rescue,"harm":harm,"changed":len(ts)})
        report["rankings"][metric]=vals
    report["candidates"].sort(key=lambda x:(x["dev_accuracy"],-x["k"]),reverse=True)
    (out/f"{a.prefix}_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"anchor_accuracy":anchor_acc,"eligible_dev":sum(r['eligible'] for r in dr),"eligible_test":sum(r['eligible'] for r in tr),"top":report['candidates'][:12]},ensure_ascii=False,indent=2))

if __name__=="__main__": main()