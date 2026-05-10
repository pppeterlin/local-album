"""
Xiaomi_Labeler.py — 隱私脫敏與雲端標註（小米 MiMo Vision）

針對 Smart_Sampler.py 採樣出的代表照片：
  1. 隱私保護：移除全部 EXIF（含 GPS、拍攝時間），重新編碼為純粹 JPEG。
  2. 等比縮放：長邊不超過 1024px。
  3. 透過小米 MiMo Vision API（OpenAI 相容介面）取得標註結果，
     輸出標準化 JSON。

API key / base url 透過建構參數或環境變數提供：
    MIMO_API_KEY        （必填）
    MIMO_BASE_URL       （可選，預設 https://api.xiaomimimo.com/v1）
    MIMO_MODEL          （可選，預設 mimo-v2.5）
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

LOGGER = logging.getLogger("XiaomiLabeler")

DEFAULT_MAX_LONG_EDGE = 1024
DEFAULT_TIMEOUT = 60
DEFAULT_JPEG_QUALITY = 90
DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_PROMPT = "请用中文描述这张图片的内容"
DEFAULT_MAX_TOKENS = 500
JPEG_MIME = "image/jpeg"


# ---------- 隱私脫敏 --------------------------------------------------------

class PrivacyProcessor:
    """移除 EXIF 並等比縮放至長邊 ≤ max_long_edge。輸出 JPEG bytes。"""

    def __init__(
        self,
        max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        if max_long_edge <= 0:
            raise ValueError("max_long_edge must be > 0")
        self.max_long_edge = int(max_long_edge)
        self.jpeg_quality = int(jpeg_quality)

    def process(self, src_path: os.PathLike | str) -> bytes:
        src_path = Path(src_path)
        with Image.open(src_path) as img:
            img.load()
            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:  # noqa: BLE001
                pass
            img = img.convert("RGB")

            w, h = img.size
            longest = max(w, h)
            if longest > self.max_long_edge:
                scale = self.max_long_edge / float(longest)
                new_size = (
                    max(1, int(round(w * scale))),
                    max(1, int(round(h * scale))),
                )
                img = img.resize(new_size, Image.LANCZOS)

            # 透過新建 Image 物件徹底剝除 EXIF / ICC / 其他 metadata
            clean = Image.new("RGB", img.size)
            clean.paste(img)
            buf = io.BytesIO()
            clean.save(
                buf,
                format="JPEG",
                quality=self.jpeg_quality,
                optimize=True,
                progressive=False,
            )
            return buf.getvalue()

    @staticmethod
    def to_data_url(image_bytes: bytes, mime: str = JPEG_MIME) -> str:
        """文檔要求格式：data:{MIME_TYPE};base64,$BASE64_IMAGE"""
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"


# ---------- API 客戶端（OpenAI 相容） --------------------------------------

class XiaomiVisionClient:
    """以 OpenAI SDK 呼叫小米 MiMo 多模態 API。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
        extra_body: Optional[Dict] = None,
    ):
        try:
            from openai import OpenAI  # 延遲匯入，避免無 openai 環境時 import 即失敗
        except ImportError as e:  # noqa: BLE001
            raise ImportError(
                "請先安裝 openai 套件：pip install openai>=1.0"
            ) from e

        self.api_key = api_key or os.environ.get("MIMO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "缺少 MIMO_API_KEY。請傳入 api_key 或設定環境變數 MIMO_API_KEY。"
            )
        self.base_url = base_url or os.environ.get("MIMO_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("MIMO_MODEL", DEFAULT_MODEL)
        self.timeout = int(timeout)
        self.max_retries = max(1, int(max_retries))
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.extra_body = extra_body

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @staticmethod
    def _default_system_prompt() -> str:
        today = datetime.now().strftime("%A, %B %d, %Y")
        return (
            f"You are MiMo, an AI assistant developed by Xiaomi. "
            f"Today is date: {today}. Your knowledge cutoff date is December 2024."
        )

    def label(
        self,
        image_data_urls: List[str],
        prompt: str = DEFAULT_PROMPT,
        max_completion_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Dict:
        """傳入一或多張圖片（data URL 形式），回傳完整 chat completion JSON。"""
        if not image_data_urls:
            raise ValueError("image_data_urls 至少需 1 張")

        user_content: List[Dict] = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_data_urls
        ]
        user_content.append({"type": "text", "text": prompt})

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=max_completion_tokens,
            extra_body=self.extra_body,
        )
        # OpenAI SDK 物件 → dict
        return json.loads(completion.model_dump_json())

    @staticmethod
    def extract_text(response: Dict) -> str:
        """從 chat completion JSON 中取出 assistant 第一條訊息純文字。"""
        try:
            choices = response.get("choices") or []
            if not choices:
                return ""
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # 多段內容（少見）：取出 type=text 的部分串接
                return "".join(
                    seg.get("text", "")
                    for seg in content
                    if isinstance(seg, dict) and seg.get("type") == "text"
                )
            return ""
        except Exception:  # noqa: BLE001
            return ""


# ---------- 主類別 ----------------------------------------------------------

class XiaomiLabeler:
    def __init__(
        self,
        client: Optional[XiaomiVisionClient] = None,
        privacy: Optional[PrivacyProcessor] = None,
        prompt: str = DEFAULT_PROMPT,
        max_completion_tokens: int = DEFAULT_MAX_TOKENS,
        concurrency: int = 1,
    ):
        self.client = client or XiaomiVisionClient()
        self.privacy = privacy or PrivacyProcessor()
        self.prompt = prompt
        self.max_completion_tokens = int(max_completion_tokens)
        self.concurrency = max(1, int(concurrency))

    def label_image(self, path: os.PathLike | str) -> Dict:
        path = Path(path)
        result: Dict = {"path": str(path)}
        try:
            clean_bytes = self.privacy.process(path)
            result["bytes"] = len(clean_bytes)
            data_url = self.privacy.to_data_url(clean_bytes, mime=JPEG_MIME)
        except Exception as e:  # noqa: BLE001
            LOGGER.error("Privacy preprocess failed for %s: %s", path, e)
            result["error"] = f"preprocess: {e}"
            return result
        try:
            raw = self.client.label(
                [data_url],
                prompt=self.prompt,
                max_completion_tokens=self.max_completion_tokens,
            )
            result["response"] = raw
            result["text"] = self.client.extract_text(raw)
        except Exception as e:  # noqa: BLE001
            LOGGER.error("API call failed for %s: %s", path, e)
            result["error"] = f"api: {e}"
        return result

    def label_many(self, paths: Iterable[os.PathLike | str]) -> List[Dict]:
        results: List[Dict] = []
        for i, p in enumerate(paths, 1):
            LOGGER.info("[%d] labeling %s", i, p)
            results.append(self.label_image(p))
        return results

    def label_from_samples(
        self,
        samples_json: os.PathLike | str,
        output_path: os.PathLike | str = "labels.json",
        incremental: bool = False,
    ) -> Dict:
        data = json.loads(Path(samples_json).read_text(encoding="utf-8"))
        paths: List[str] = (
            data.get("paths")
            or [s["path"] for s in data.get("samples", [])]
        )
        if not paths:
            LOGGER.warning("No paths in samples; nothing to label.")

        # 增量模式：載入既有 labels，跳過已標注的圖片
        existing_results: List[Dict] = []
        labeled_paths: set = set()
        if incremental:
            out = Path(output_path)
            if out.exists():
                existing = json.loads(out.read_text(encoding="utf-8"))
                existing_results = existing.get("results", [])
                labeled_paths = {r["path"] for r in existing_results if "error" not in r}
                LOGGER.info("Incremental: found %d existing labels, %d succeeded", len(existing_results), len(labeled_paths))

        # 篩選出需要標注的新圖片
        new_paths = [p for p in paths if p not in labeled_paths]
        if incremental and labeled_paths:
            LOGGER.info("New images to label: %d (skipping %d existing)", len(new_paths), len(labeled_paths))

        if not new_paths:
            LOGGER.info("No new images to label — keeping existing labels")
            return {
                "model": self.client.model,
                "count": len(existing_results),
                "succeeded": sum(1 for r in existing_results if "error" not in r),
                "failed": sum(1 for r in existing_results if "error" in r),
                "results": existing_results,
            }

        # 並行標注
        from concurrent.futures import ThreadPoolExecutor, as_completed

        new_results: List[Dict] = []
        batch_size = self.concurrency * 5  # 每批存檔一次

        LOGGER.info("Starting labeling: %d images, concurrency=%d, batch_size=%d",
                     len(new_paths), self.concurrency, batch_size)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # 分批處理
            for batch_start in range(0, len(new_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(new_paths))
                batch_paths = new_paths[batch_start:batch_end]

                # 提交整批任務
                future_to_path = {
                    executor.submit(self.label_image, p): (batch_start + i, p)
                    for i, p in enumerate(batch_paths)
                }

                # 收集結果（保持順序）
                batch_results: Dict[int, Dict] = {}
                for future in as_completed(future_to_path):
                    idx, path = future_to_path[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        LOGGER.error("Label failed for %s: %s", path, e)
                        result = {"path": path, "error": str(e)}
                    batch_results[idx] = result
                    status = "ok" if "error" not in result else "fail"
                    LOGGER.info("[%d/%d] %s → %s", idx + 1, len(new_paths), Path(path).name, status)

                # 按順序加入結果
                for idx in sorted(batch_results.keys()):
                    new_results.append(batch_results[idx])

                # 每批存檔
                all_results = existing_results + new_results
                succeeded = sum(1 for r in all_results if "error" not in r)
                payload = {
                    "model": self.client.model,
                    "count": len(all_results),
                    "succeeded": succeeded,
                    "failed": len(all_results) - succeeded,
                    "results": all_results,
                }
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                LOGGER.info("Batch saved: %d/%d done (%d ok, %d fail)",
                           len(new_results), len(new_paths), succeeded, len(all_results) - succeeded)

        # 最終統計
        all_results = existing_results + new_results
        succeeded = sum(1 for r in all_results if "error" not in r)
        payload = {
            "model": self.client.model,
            "count": len(all_results),
            "succeeded": succeeded,
            "failed": len(all_results) - succeeded,
            "results": all_results,
        }
        LOGGER.info(
            "Done: %d total labels (%d ok, %d fail) → %s",
            payload["count"], payload["succeeded"], payload["failed"], output_path,
        )
        return payload


# ---------- CLI -------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="小米 MiMo Vision 標註（含隱私脫敏前處理）")
    p.add_argument("samples", help="Smart_Sampler 產出的 samples.json")
    p.add_argument("-o", "--output", default="labels.json")
    p.add_argument("--api-key", default=None, help="覆蓋 MIMO_API_KEY")
    p.add_argument("--base-url", default=None, help=f"預設 {DEFAULT_BASE_URL}")
    p.add_argument("--model", default=None, help=f"預設 {DEFAULT_MODEL}")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--max-long-edge", type=int, default=DEFAULT_MAX_LONG_EDGE)
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--no-reasoning", action="store_true", help="關閉 reasoning/thinking tokens，節省 token 用量")
    p.add_argument("--incremental", action="store_true", help="增量標注：跳過已標注圖片，每批存檔")
    p.add_argument("--concurrency", type=int, default=5, help="並行標注數（預設 5，100 RPM 下建議 5-10）")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    _setup_logging(args.log_level)

    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_reasoning else None
    client = XiaomiVisionClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        extra_body=extra_body,
    )
    privacy = PrivacyProcessor(
        max_long_edge=args.max_long_edge,
        jpeg_quality=args.jpeg_quality,
    )
    labeler = XiaomiLabeler(
        client=client,
        privacy=privacy,
        prompt=args.prompt,
        max_completion_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )
    labeler.label_from_samples(args.samples, args.output, incremental=args.incremental)
    return 0


if __name__ == "__main__":
    sys.exit(main())
