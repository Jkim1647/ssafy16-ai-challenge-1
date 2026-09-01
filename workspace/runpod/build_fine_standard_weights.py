from __future__ import annotations
import argparse,csv,json
from pathlib import Path
CHOICES="abcd"; WEIGHTS=[0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54]
def read(p):
    with Path(p).open(encoding="utf-8") as f:return {r["id"]:r for r in map(json.loads,filter(str.strip,f))}
def prob(r):return [float(r[f"score_{x}"]) for x in CHOICES]
def label(v):return CHOICES[max(range(4),key=v.__getitem__)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--base-dev",nargs=2,required=True);ap.add_argument("--new-dev",required=True);ap.add_argument("--base-test",nargs=2,required=True);ap.add_argument("--new-test",required=True);ap.add_argument("--sample",required=True);ap.add_argument("--output-dir",required=True);a=ap.parse_args()
    bd=list(map(read,a.base_dev));nd=read(a.new_dev);bt=list(map(read,a.base_test));nt=read(a.new_test);dids=list(nd);tids=[r["id"] for r in csv.DictReader(open(a.sample,encoding="utf-8-sig"))];out=Path(a.output_dir);report=[]
    anchor={}
    for w in WEIGHTS:
        good=0;ans=[]
        for i in dids:
            b=[sum(prob(x[i])[j] for x in bd)/2 for j in range(4)];n=prob(nd[i]);z=label([(1-w)*b[j]+w*n[j] for j in range(4)]);good+=z==nd[i]["true_label"]
        for i in tids:
            b=[sum(prob(x[i])[j] for x in bt)/2 for j in range(4)];n=prob(nt[i]);ans.append(label([(1-w)*b[j]+w*n[j] for j in range(4)]))
        name=f"submission_qwen35_public_fine_w{w:.2f}".replace(".","p")+".csv"
        with (out/name).open("w",encoding="utf-8-sig",newline="") as f:q=csv.writer(f);q.writerow(["id","answer"]);q.writerows(zip(tids,ans))
        if w==.5:anchor=dict(zip(tids,ans))
        report.append({"file":name,"weight":w,"dev_accuracy":good/len(dids),"answers":dict(zip(tids,ans))})
    for r in report:r["changed_vs_w050"]=sum(v!=r["answers"][k] for k,v in anchor.items());del r["answers"]
    (out/"qwen35_public_fine_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()