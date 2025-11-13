import json
from pathlib import Path

def jsonl_to_txt(folder: str):
    base = Path(folder)
    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    for jsonl_path in sorted(base.glob("*.jsonl")):
        txt_path = jsonl_path.with_suffix(".txt")
        with jsonl_path.open("r", encoding="utf-8") as fin, txt_path.open("w", encoding="utf-8") as fout:
            for idx, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    # 如果遇到坏行，跳过（也可以选择抛出异常）
                    print(f"[WARN] {jsonl_path.name} 第{idx}行 JSON 解析失败：{e}")
                    continue

                q = (obj.get("question") or "").replace("\r\n", "\n").strip()
                a = (obj.get("answer") or "").strip()

                # 写入
                fout.write(f"# Question {idx}\n")
                fout.write(f"{q}\n")
                fout.write(f"answer: {a}\n")
                # 题与题之间空一行
                fout.write("\n")

        print(f"[OK] Wrote: {txt_path}")
if __name__ == "__main__":
    jsonl_to_txt('/home/yz54720/Projects/Method/deepconf/data/raw')