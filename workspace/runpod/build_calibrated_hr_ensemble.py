from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path

CHOICES = "abcd"
GAMMAS = [0.25, 0.4, 0.6, 0.8, 1.0, 1.25]
NEW_MIXES = [[0, 0, 1], [0, 0.5, 0.5], [0.2, 0.4, 0.4], [1/3, 1/3, 1/3]]

def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}

def raw(row): return [float(row[f"score_{x}"]) for x in CHOICES]
def calibrate(p, gamma):
    q = [max(x, 1e-12) ** gamma for x in p]; s = sum(q)
    return [x/s for x in q]
def choose(p): return CHOICES[max(range(4), key=p.__getitem__)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-dev",nargs=2,required=True); ap.add_argument("--new-dev",nargs=3,required=True)
    ap.add_argument("--base-test",nargs=2,required=True); ap.add_argument("--new-test",nargs=3,required=True)
    ap.add_argument("--sample",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--prefix",default="qwen35_calhr")
    a=ap.parse_args(); bd=list(map(read,a.base_dev)); nd=list(map(read,a.new_dev)); bt=list(map(read,a.base_test)); nt=list(map(read,a.new_test))
    dids=list(bd[0]); tids=[r["id"] for r in csv.DictReader(open(a.sample,encoding="utf-8-sig"))]
    if any(set(x)!=set(dids) for x in bd+nd) or any(set(x)!=set(tids) for x in bt+nt): raise ValueError("ID mismatch")
    truth=[nd[0][i]["true_label"] for i in dids]
    configs=[]
    for gb in GAMMAS:
      for gn in GAMMAS:
       db=[[[*calibrate(raw(x[i]),gb)] for x in bd] for i in dids]
       dn=[[[*calibrate(raw(x[i]),gn)] for x in nd] for i in dids]
       for mix in NEW_MIXES:
        for tick in range(12,29):
         wb=tick/40; wn=1-wb; preds=[]; fc=[0]*4; ft=[0]*4
         for i,b,n,y in zip(dids,db,dn,truth):
          bp=[sum(x[j] for x in b)/2 for j in range(4)]
          np=[sum(mix[k]*n[k][j] for k in range(3)) for j in range(4)]
          z=choose([wb*bp[j]+wn*np[j] for j in range(4)]); preds.append(z)
          f=int(hashlib.md5(i.encode()).hexdigest(),16)%4; ft[f]+=1; fc[f]+=z==y
         folds=[x/y for x,y in zip(fc,ft)]
         configs.append({"accuracy":sum(x==y for x,y in zip(preds,truth))/len(truth),"fold_min":min(folds),"folds":folds,"base_gamma":gb,"new_gamma":gn,"base_weight":wb,"new_mix":mix})
    configs.sort(key=lambda x:(x["accuracy"],x["fold_min"]),reverse=True); best=configs[0]["accuracy"]
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); seen=set(); made=[]
    for cfg in configs:
      if cfg["accuracy"] < best-2/len(dids): break
      gb,gn,wb,mix=cfg["base_gamma"],cfg["new_gamma"],cfg["base_weight"],cfg["new_mix"]; wn=1-wb; ans=[]
      for i in tids:
       bp=[sum(calibrate(raw(x[i]),gb)[j] for x in bt)/2 for j in range(4)]
       np=[sum(mix[k]*calibrate(raw(nt[k][i]),gn)[j] for k in range(3)) for j in range(4)]
       ans.append(choose([wb*bp[j]+wn*np[j] for j in range(4)]))
      key=tuple(ans)
      if key in seen: continue
      seen.add(key); name=f"submission_{a.prefix}_rank{len(made)+1:02d}.csv"
      with (out/name).open("w",encoding="utf-8-sig",newline="") as f:
       w=csv.writer(f); w.writerow(["id","answer"]); w.writerows(zip(tids,ans))
      made.append({"file":name,**cfg})
      if len(made)>=20: break
    report={"dev_samples":len(dids),"grid_size":len(configs),"best_accuracy":best,"candidates":made,"top_grid":configs[:60]}
    (out/f"{a.prefix}_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"best_accuracy":best,"generated":len(made),"top":made[:6]},ensure_ascii=False,indent=2))

if __name__=="__main__": main()