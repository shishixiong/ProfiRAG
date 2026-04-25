import html
import base64
from markdownify import MarkdownConverter

def decode_html_entities(text: str) -> str:
    """处理 &#x...; 或 &#...; 的实体解码"""
    return html.unescape(text)

class WikiMarkdownConverter(MarkdownConverter):
    """自定义Markdown转换器，支持华为Wiki特殊代码块格式"""

    def convert_pre(self, el, text, parent_tags):
        """处理常规代码块，从class属性提取语言信息"""
        if not text:
            return ''

        code_language = ''
        classes = el.get('class', [])
        if isinstance(classes, str):
            classes = classes.split()
        for cls in classes:
            if cls.startswith('language-'):
                code_language = cls.replace('language-', '')
                break

        if self.options['strip_pre'] == 'strip':
            text = text.strip()
        elif self.options['strip_pre'] == 'strip_one':
            text = text.strip('\n')

        return '\n\n```%s\n%s\n```\n\n' % (code_language, text)

    def convert_ce_monaco_code_editor(self, el, text, parent_tags):
        """处理Monaco编辑器代码块，解码base64内容"""
        language = el.get('language', '')
        code_b64 = el.get('code', '')

        if code_b64:
            try:
                code = base64.b64decode(code_b64).decode('utf-8')
            except Exception:
                code = code_b64
        else:
            code = text

        return '\n\n```%s\n%s\n```\n\n' % (language, code)

def html_to_markdown(html_content: str) -> str:
    """将HTML转换为Markdown，支持华为Wiki特殊代码块格式"""
    converter = WikiMarkdownConverter(
        heading_style="ATX",
        bullets="-",
        strip_pre='strip',
        table_infer_header=True
    )
    return converter.convert(html_content)