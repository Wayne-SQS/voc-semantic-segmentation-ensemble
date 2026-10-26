# -*- coding: utf-8 -*-
"""把 pandoc 生成的报告 docx 规范成学术排版（可重复运行）。

处理项：
  1. 正文：宋体 + Times New Roman 12pt（小四），1.5 倍行距，首行缩进 2 字符
  2. 标题：H1 黑体三号居中 / H2 黑体四号 / H3 黑体小四（拉开层级）
  3. 全部表格改三线表：上下粗线 1.5pt、表头下细线 0.75pt、无竖线
  4. 图表题注：宋体五号（10.5pt）居中
  5. 图片段落居中

用法：
    python polish_docx.py --src ..\report_孙秋实_20245715.docx --out ..\report_polished.docx
"""
import argparse

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


def apply_run_font(run, ascii_font, ea_font, size, bold=None):
    """直接改 run 的字体（这个 docx 没有 Normal/Caption 等样式，只能用直接格式）。"""
    run.font.name = ascii_font
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    for k in ("w:ascii", "w:hAnsi", "w:cs"):
        rFonts.set(qn(k), ascii_font)
    rFonts.set(qn("w:eastAsia"), ea_font)


def three_line(table):
    """三线表：顶/底 1.5pt，表头下 0.75pt，其余无边框。"""
    tbl = table._tbl
    tblPr = tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge, sz, val in (("top", 12, "single"), ("bottom", 12, "single"),
                          ("left", 0, "nil"), ("right", 0, "nil"),
                          ("insideH", 0, "nil"), ("insideV", 0, "nil")):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.insert_element_before(
        borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
        "w:tblCaption", "w:tblDescription")

    # 表头行下细线
    if table.rows:
        for cell in table.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for old in tcPr.findall(qn("w:tcBorders")):
                tcPr.remove(old)
            tcb = OxmlElement("w:tcBorders")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:space"), "0")
            bottom.set(qn("w:color"), "000000")
            tcb.append(bottom)
            tcPr.insert_element_before(tcb, "w:shd", "w:noWrap", "w:tcMar",
                                       "w:textDirection", "w:tcFitText",
                                       "w:vAlign", "w:hideMark")


def strip_marks(d):
    """去掉 Word 里在页边显示"小黑方块/黑竖线"的段落属性。

    keepNext（与下段同页）、keepLines（段中不分页）、pageBreakBefore（段前分页）
    都会让 Word 在段落左侧画一个黑色小方块；pBdr 则是 pandoc 把 markdown 列表项
    渲染成的左侧黑竖线。这些标记只影响屏幕显示、不打印，但观感很差，全部清掉。
    """
    tags = ("w:pBdr", "w:keepNext", "w:keepLines", "w:pageBreakBefore")
    n = 0
    for p in d.paragraphs:
        pPr = p._p.pPr
        if pPr is None:
            continue
        for tag in tags:
            el = pPr.find(qn(tag))
            if el is not None:
                pPr.remove(el)
                n += 1
    for st in d.styles:  # 标题样式里也可能带 keepNext，一并清掉
        pPr = getattr(st.element, "pPr", None)
        if pPr is None:
            continue
        for tag in tags:
            el = pPr.find(qn(tag))
            if el is not None:
                pPr.remove(el)
                n += 1
    return n


def polish_paragraph(para, size=12.0, indent=True, align=None, bold=None,
                     east_asia="宋体", space_before=0, space_after=3, line=1.75,
                     wrap_latin=False):
    pf = para.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.left_indent = Pt(0)
    pf.right_indent = Pt(0)
    pf.first_line_indent = Pt(size * 2) if indent else Pt(0)
    if align is not None:
        para.alignment = align
    if wrap_latin:
        # 允许西文在单词中间换行（等价 Word「允许西文在单词中间换行」），
        # 否则遇到 SegmentationClass/*.png 这类长 token 时两端对齐会把整行拉散。
        pPr = para._p.get_or_add_pPr()
        for old in pPr.findall(qn("w:wordWrap")):
            pPr.remove(old)
        el = OxmlElement("w:wordWrap")
        el.set(qn("w:val"), "0")
        pPr.insert_element_before(
            el, "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN",
            "w:bidi", "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
            "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
            "w:textDirection", "w:textAlignment", "w:textboxTightWrap",
            "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange")
    for r in para.runs:
        apply_run_font(r, "Times New Roman", east_asia, size, bold=bold)


def add_page_footer(d):
    """页脚居中页码（PAGE / NUMPAGES 域），宋体小五。"""
    footer = d.sections[0].footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0

    def _fld(instr):
        r = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        rFonts = OxmlElement("w:rFonts")
        for k in ("w:ascii", "w:hAnsi", "w:cs"):
            rFonts.set(qn(k), "Times New Roman")
        rFonts.set(qn("w:eastAsia"), "宋体")
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "18")  # 小五 = 9pt
        rPr.append(rFonts)
        rPr.append(sz)
        r.append(rPr)
        f1 = OxmlElement("w:fldChar")
        f1.set(qn("w:fldCharType"), "begin")
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = instr
        f2 = OxmlElement("w:fldChar")
        f2.set(qn("w:fldCharType"), "end")
        r.append(f1)
        r.append(it)
        r.append(f2)
        return r

    r_pre = p.add_run("— ")
    apply_run_font(r_pre, "Times New Roman", "宋体", 9)
    p._p.append(_fld(" PAGE "))
    r_sep = p.add_run(" —")
    apply_run_font(r_sep, "Times New Roman", "宋体", 9)


def strip_empty_paragraphs(d):
    """删除 pandoc 渲染遗留的 Normal 空段落。

    md 里 `---` 水平线、标题前空行、图/表后的空行，pandoc 会生成为
    style=Normal、text=''、不含图片/域代码的 <w:p/> 段落。
    这些段落本身不占空间，但会叠加相邻标题的 space_before/after，
    形成视觉上的"大段空行"。统统删掉，标题/正文的间距由各自 space
    参数保证足够的视觉分隔。
    """
    n = 0
    ns_drawing = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
    for p in list(d.paragraphs):
        txt = p.text.strip()
        has_img = bool(p._element.findall(f".//{ns_drawing}inline"))
        has_field = bool(p._element.findall(".//" + qn("w:fldChar")))
        if not txt and not has_img and not has_field:
            p._element.getparent().remove(p._element)
            n += 1
    return n


def blacken_headings(d):
    """标题样式的颜色统一改为纯黑（pandoc 默认用 Word 的 accent1 蓝 #4F81BD）。"""
    n = 0
    for st in d.styles.element.findall(qn("w:style")):
        sid = st.get(qn("w:styleId")) or ""
        if not (sid.startswith("Heading") or sid in ("Title", "Subtitle")):
            continue
        rPr = st.find(qn("w:rPr"))
        if rPr is None:
            continue
        col = rPr.find(qn("w:color"))
        if col is None:
            continue
        col.set(qn("w:val"), "000000")
        for attr in ("w:themeColor", "w:themeShade", "w:themeTint"):
            if col.get(qn(attr)) is not None:
                del col.attrib[qn(attr)]
        n += 1
    return n


def flatten_lists(d):
    """把 Word 自动编号/项目符号列表转成普通段落，去掉一切圆点。

    连续的列表项改用「（1）（2）…」编号（学术报告常用），单项列表直接并入正文；
    列表自带的左缩进与悬挂缩进一并清除，改由正文的首行缩进统一控制。
    """
    groups, cur = [], []
    for p in d.paragraphs:
        st = p.style.name if p.style is not None else ""
        pPr = p._p.pPr
        numbered = pPr is not None and pPr.find(qn("w:numPr")) is not None
        if st == "Compact" or numbered:
            cur.append(p)
        else:
            if cur:
                groups.append(cur)
                cur = []
    if cur:
        groups.append(cur)

    for g in groups:
        for i, p in enumerate(g, 1):
            pPr = p._p.pPr
            if pPr is not None:
                for tag in ("w:numPr", "w:ind"):
                    el = pPr.find(qn(tag))
                    if el is not None:
                        pPr.remove(el)
            if len(g) < 2:
                continue
            run = p.add_run(f"（{i}）")
            apply_run_font(run, "Times New Roman", "宋体", 12, bold=False)
            el = run._element
            el.getparent().remove(el)
            p._p.insert(1 if p._p.pPr is not None else 0, el)
    return len(groups)


def soften_long_tokens(d):
    """在西文长 token 的 "/" 后插入零宽空格，给 Word 一个合法断行点。

    技术报告里 JPEGImages/*.jpg、SegmentationClass/*.png 这类 token 无法断行，
    两端对齐时会把整行汉字间距拉开；有了零宽断点，换行位置更合理、字距更均匀。
    """
    import re as _re
    n = 0
    for p in d.paragraphs:
        st = p.style.name if p.style is not None else ""
        if st.startswith("Heading") or st.startswith("toc"):
            continue
        for r in p.runs:
            t = r.text
            if "/" not in t:
                continue
            new = _re.sub(r"/(?=[^\s\u200b])", "/\u200b", t)
            if new != t:
                r.text = new
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Document(args.src)

    groups = flatten_lists(d)

    # 段落层（该 docx 只有 Heading/toc 样式，正文无样式，只能逐段设直接格式）
    for p in d.paragraphs:
        st = p.style.name if p.style is not None else ""
        txt = p.text.strip()
        if not txt or st.startswith("toc"):
            continue
        if st == "Heading 1":
            polish_paragraph(p, 16, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER,
                             bold=True, east_asia="黑体", space_before=24, space_after=10,
                             line=1.3)
        elif st == "Heading 2":
            polish_paragraph(p, 14, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT,
                             bold=True, east_asia="黑体", space_before=16, space_after=8,
                             line=1.3)
        elif st == "Heading 3":
            polish_paragraph(p, 12, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT,
                             bold=True, east_asia="黑体", space_before=12, space_after=6,
                             line=1.3)
        elif txt.startswith(("图 ", "表 ")):
            polish_paragraph(p, 10.5, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER,
                             bold=False, space_before=4, space_after=10)
        elif p._element.findall(".//" + qn("w:drawing")):
            polish_paragraph(p, 10.5, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER,
                             space_before=6, space_after=4)
        else:
            polish_paragraph(p, 12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.6,
                             space_after=4)

    for t in d.tables:
        three_line(t)
    # 紧跟表格的段落补一点段前距，避免正文贴住表底细线
    body = d.element.body
    for i, el in enumerate(list(body)):
        if i and body[i - 1].tag == qn("w:tbl") and el.tag == qn("w:p"):
            from docx.text.paragraph import Paragraph
            pp = Paragraph(el, d)
            if pp.text.strip():
                pp.paragraph_format.space_before = Pt(8)

    removed = strip_marks(d)
    removed += strip_empty_paragraphs(d)

    # 表题与下方表格、图片与下方图题绑在一起，避免题注被甩到页尾
    for p in d.paragraphs:
        t = p.text.strip()
        st = p.style.name if p.style is not None else ""
        if (t.startswith("表 ") or st.startswith("Heading")
                or p._element.findall(".//" + qn("w:drawing"))):
            p.paragraph_format.keep_with_next = True

    blackened = blacken_headings(d)
    zw = soften_long_tokens(d)
    add_page_footer(d)

    d.save(args.out)
    print(f"已输出：{args.out}（段落 {len(d.paragraphs)}，表格 {len(d.tables)}，"
          f"列表去圆点 {groups} 组，清除黑方块/边线标记 {removed} 处，"
          f"标题改黑 {blackened} 处，长 token 加断点 {zw} 处，已加页脚页码）")


if __name__ == "__main__":
    main()
