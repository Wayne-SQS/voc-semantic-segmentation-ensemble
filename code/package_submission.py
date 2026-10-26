# -*- coding: utf-8 -*-
"""一键打包最终提交物（作业 P5 提交要求）。

作业原文（PDF 第 5 页）：
    ⚫ A project report (e.g. report_studentname_studentId.docx)
    ⚫ Your source code file. (e.g. code_studentname_studentid)
    ⚫ The prediction of test set. (e.g. predict_studentname_studentid.zip)
    ⚫ Zip these three files into a folder named studentname_studentid

本脚本产出：
    孙秋实_20245715/
        report_孙秋实_20245715.docx
        code_孙秋实_20245715/            （源码，剔除 runs / outputs / __pycache__）
        predict_孙秋实_20245715.zip      （400 张单通道灰度 mask，平铺在 zip 根目录）
    -> 再整体压缩为  孙秋实_20245715.zip

用法（报告还没写好时可以先不带 --report，脚本会提示缺失）：
    python package_submission.py --mask_dir ..\outputs\predict
"""

import argparse
import os
import shutil
import zipfile

STUDENT_NAME = "孙秋实"
STUDENT_ID = "20245715"

SKIP_DIRS = {"runs", "outputs", "__pycache__", ".git", ".idea", "checkpoints"}
SKIP_EXT = {".pt", ".pth", ".zip", ".log"}


def copy_code(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, src_dir)
        target_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(target_root, exist_ok=True)
        for f in files:
            if os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            shutil.copy2(os.path.join(root, f), os.path.join(target_root, f))
            n += 1
    return n


def zip_masks(mask_dir, zip_path):
    names = sorted([n for n in os.listdir(mask_dir) if n.lower().endswith(".png")])
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(mask_dir, n), arcname=n)
    return len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--mask_dir", default=r"E:\桌面\人工智能技术大作业\outputs\predict")
    ap.add_argument("--report", default="", help="报告 docx 路径，可后续再补")
    ap.add_argument("--out_root", default=r"E:\桌面\人工智能技术大作业\outputs\submission")
    args = ap.parse_args()

    folder = f"{STUDENT_NAME}_{STUDENT_ID}"
    work = os.path.join(args.out_root, folder)
    if os.path.isdir(work):
        shutil.rmtree(work)  # 重跑时覆盖（只删自己刚生成的目录）
    os.makedirs(work, exist_ok=True)

    # 1) 报告
    report_dst = os.path.join(work, f"report_{STUDENT_NAME}_{STUDENT_ID}.docx")
    if args.report and os.path.isfile(args.report):
        shutil.copy2(args.report, report_dst)
        report_state = "已放入"
    else:
        report_state = "⚠ 缺失（报告定稿后补 --report 参数重跑）"

    # 2) 源码
    code_dst = os.path.join(work, f"code_{STUDENT_NAME}_{STUDENT_ID}")
    n_code = copy_code(args.code_dir, code_dst)

    # 3) 预测
    zip_path = os.path.join(work, f"predict_{STUDENT_NAME}_{STUDENT_ID}.zip")
    n_mask = zip_masks(args.mask_dir, zip_path)

    # 4) 外层压缩
    outer = os.path.join(args.out_root, folder + ".zip")
    if os.path.exists(outer):
        os.remove(outer)
    with zipfile.ZipFile(outer, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(work):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, arcname=os.path.join(folder, os.path.relpath(p, work)))

    print(f"外层压缩包: {outer}  ({os.path.getsize(outer) / 1e6:.1f} MB)")
    print(f"  - report : {report_state}")
    print(f"  - code   : {n_code} 个文件")
    print(f"  - predict: {n_mask} 张 mask")
    print("\n提交前请再确认：邮件发往 2940986065@qq.com，截止 11 月 15 日（建议 11/13 前发出）。")


if __name__ == "__main__":
    main()
