"""简历 PDF 解析工具。

职责：把上传的简历 PDF 内容抽成纯文本，供下游（结构化解析、ResumeAgent、
岗位匹配、面试）复用——它们都读 resume.raw_text，所以只需在"上传"这一步
完成 PDF→文本转换即可。

策略（两级回退）：
1. 直接提取：用 PyMuPDF(fitz) 逐页 get_text()，适合电子版（文字可选中的）PDF。
2. OCR 兜底：当提取到的文本过少（判定为扫描件）时，把每页渲染成图片交给
   OCR 识别。OCR 依赖系统安装 Tesseract-OCR + 中文语言包 chi_sim；没装时
   抛出 PDFOcrUnavailableError，由接口给出清晰提示。

错误契约：
- PDFParseError          ：不是合法 PDF / 解析失败 / 未装解析库。
- PDFOcrUnavailableError ：扫描件但没有可用的 OCR（缺库或缺 Tesseract）。
"""

import io

# 提取文本低于该长度时判定为扫描件，触发 OCR 兜底
OCR_MIN_TEXT = 20


class PDFParseError(Exception):
    """PDF 无法被解析（非合法 PDF / 解析库缺失）。"""


class PDFOcrUnavailableError(Exception):
    """扫描件但没有可用的 OCR（缺 pytesseract / Tesseract / 语言包）。"""


def extract_pdf_text(data: bytes, filename: str = "") -> str:
    """把 PDF 字节内容解析成纯文本（直接提取，必要时 OCR 兜底）。

    参数:
        data: PDF 文件的二进制内容。
        filename: 文件名，仅用于错误信息提示。

    返回:
        解析出的文本（去首尾空白）。

    抛出:
        PDFParseError / PDFOcrUnavailableError。
    """
    try:
        import pymupdf as fitz  # PyMuPDF 新版 API
    except ImportError:
        raise PDFParseError("未安装 pymupdf，无法解析 PDF。请运行：pip install pymupdf")

    # 从字节流在内存打开 PDF，避免写临时文件
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFParseError(f"无法解析该文件（{filename or '未知文件'} 不是合法 PDF）：{exc}")

    text = ""
    try:
        for page in doc:
            text += page.get_text()
    finally:
        doc.close()

    text = text.strip()
    # 提取到的文本太少 -> 很可能是扫描件（无文字层），走 OCR 兜底
    if len(text) < OCR_MIN_TEXT:
        ocr_text = _ocr_pdf(data)
        if ocr_text:
            return ocr_text
        raise PDFOcrUnavailableError(
            "该 PDF 没有可提取的文字层（疑似扫描件），但当前未配置 OCR。"
            "请安装 Tesseract-OCR 及中文语言包 chi_sim，或改用文本型 PDF。"
        )
    return text


def _ocr_pdf(data: bytes) -> str:
    """把扫描件 PDF 的每一页渲染成图片，再用 OCR 识别为文本。"""
    try:
        import pymupdf as fitz  # PyMuPDF 新版 API
    except ImportError:
        raise PDFOcrUnavailableError("未安装 pymupdf，无法渲染扫描件以进行 OCR。")

    try:
        import pytesseract
    except ImportError:
        raise PDFOcrUnavailableError("未安装 pytesseract，无法执行 OCR。请运行：pip install pytesseract")

    # pytesseract 调用时会抛 TesseractNotFoundError；无法在导入时可靠检测，
    # 因此在第一次实际识别时统一捕获，翻译成明确提示。
    parts: list[str] = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for page in doc:
                # 渲染成 200dpi 位图，再转成 PNG 字节交给 OCR
                pix = page.get_pixmap(dpi=200)
                png_bytes = pix.tobytes("png")
                # pytesseract 接收 PIL Image，这里从内存字节打开
                image = __import__("PIL.Image", fromlist=["Image"]).open(io.BytesIO(png_bytes))
                # 中英双语识别：需要 Tesseract 安装了 chi_sim 语言包
                text = pytesseract.image_to_string(image, lang="chi_sim+eng")
                if text.strip():
                    parts.append(text.strip())
        finally:
            doc.close()
    except Exception:
        raise PDFOcrUnavailableError(
            "OCR 不可用：请确认服务器已安装 Tesseract-OCR，并包含中文语言包 chi_sim。"
        )
    return "\n".join(parts)