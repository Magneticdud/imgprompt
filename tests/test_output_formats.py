"""Format routing and byte preservation, with no network calls."""

import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from imgprompt.images import save_image_bytes
from imgprompt.providers.base import GenerationRequest
from imgprompt.providers.openai_provider import OpenAIProvider
from imgprompt.providers.openrouter_provider import OpenRouterProvider


def request(fmt="svg", **changes):
    values = dict(
        prompt="A bakery logo",
        model="gpt-image-2.6-flare",
        aspect_ratio="1:1",
        res_key="1024x1024",
        quality_key="low",
        extras={"output_format": fmt},
    )
    values.update(changes)
    return GenerationRequest(**values)


@pytest.mark.parametrize("tier", ["flare", "sunburst"])
def test_formats_depend_on_route_and_inputs(tier):
    direct = OpenAIProvider()
    model = f"gpt-image-2.6-{tier}"
    assert "svg" in direct.get_output_formats(model)
    assert "svg" not in direct.get_output_formats(model, True)
    router = OpenRouterProvider.__new__(OpenRouterProvider)
    assert router.get_output_formats("openai/" + model) == [
        "png",
        "jpeg",
        "webp",
        "gif",
    ]


@pytest.mark.parametrize(
    "changes",
    [{"images": ["reference.png"]}, {"n": 2}, {"model": "gpt-image-2.5-flare"}],
)
def test_invalid_svg_rejected_before_network(changes):
    with pytest.raises(ValueError):
        OpenAIProvider().validate_output(request(**changes))


def test_openrouter_svg_rejected():
    router = OpenRouterProvider.__new__(OpenRouterProvider)
    with pytest.raises(ValueError):
        router.validate_output(request(model="openai/gpt-image-2.6-flare"))


@pytest.mark.parametrize("fmt", ["svg", "gif"])
def test_openai_generate_forwards_format(fmt):
    client = MagicMock()
    req = request(fmt)
    OpenAIProvider()._generate(client, req)
    assert client.images.generate.call_args.kwargs["output_format"] == fmt
    assert req.extras == {"output_format": fmt}


def test_new_openai_fields_use_extra_body():
    kwargs = OpenAIProvider()._output_kwargs(
        request(
            "gif",
            extras={
                "output_format": "gif",
                "complexity": "low",
                "frame_count": 4,
                "frame_duration_ms": 100,
            },
        )
    )
    assert kwargs == {
        "output_format": "gif",
        "extra_body": {
            "complexity": "low",
            "frame_count": 4,
            "frame_duration_ms": 100,
        },
    }


@pytest.mark.parametrize("batch", [False, True])
def test_openai_edit_forwards_format_and_animation_fields(monkeypatch, batch):
    monkeypatch.setattr(
        "imgprompt.providers.openai_provider.process_image_for_api",
        lambda *args: ("ref.png", io.BytesIO(), "image/png"),
    )
    client = MagicMock()
    client.images.edit.return_value.data = []
    req = request(
        "gif",
        images=["a.png", "b.png"] if batch else ["a.png"],
        res_key="auto",
        extras={"output_format": "gif", "frame_count": 4},
    )
    provider = OpenAIProvider()
    if batch:
        provider._run_batch(client, req)
    else:
        provider._run_single(client, req)
    assert client.images.edit.call_count == (2 if batch else 1)
    for call in client.images.edit.call_args_list:
        assert call.kwargs["output_format"] == "gif"
        assert call.kwargs["extra_body"] == {"frame_count": 4}
        assert call.kwargs["size"] == "auto"


@pytest.mark.parametrize("tier", ["flare", "sunburst"])
def test_openrouter_gif_payload(tier):
    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    req = request(
        "gif", model=f"openai/gpt-image-2.6-{tier}", res_key="auto", aspect_ratio="Auto"
    )
    payload = provider._build_payload(req)
    assert payload["output_format"] == "gif"
    assert payload["quality"] == "low"
    assert payload["size"] == "auto"
    assert "resolution" not in payload
    assert "aspect_ratio" not in payload


def test_svg_saved_as_original_xml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svg = b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"/>'
    save_image_bytes(svg, None)
    assert next(tmp_path.glob("*.svg")).read_bytes() == svg


def test_gif_saved_with_all_frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stream = io.BytesIO()
    frames = [Image.new("RGB", (16, 16), color) for color in ("red", "blue")]
    frames[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    data = stream.getvalue()
    save_image_bytes(data, None)
    saved = next(tmp_path.glob("*.gif"))
    assert saved.read_bytes() == data
    with Image.open(saved) as gif:
        assert gif.n_frames == 2


@pytest.mark.parametrize("fmt", ["svg", "gif", "png"])
def test_wizard_persists_selected_format(monkeypatch, fmt, capsys):
    import imgedit

    saved = []
    ran = []

    class Provider(OpenAIProvider):
        def run(self, req):
            self.validate_output(req)
            ran.append(req)

    monkeypatch.setattr("sys.argv", ["imgedit.py", "--free", "--no-preview"])
    monkeypatch.setattr(imgedit, "PROVIDER_MAP", {"OpenAI": Provider})
    monkeypatch.setattr(imgedit, "step_provider", lambda: "OpenAI")
    monkeypatch.setattr(imgedit, "step_model", lambda p: "gpt-image-2.6-flare")
    monkeypatch.setattr(
        imgedit, "step_resolution", lambda *a: ("1:1", "1024x1024", 1024, 1024)
    )
    monkeypatch.setattr(imgedit, "step_quality", lambda *a: ("low", 0.0059))
    monkeypatch.setattr(imgedit, "step_prompt", lambda *a: ("A logo", None))
    monkeypatch.setattr(imgedit, "save_last_generation", lambda p, r: saved.append(r))

    def select(message, **kwargs):
        answer = fmt if message == "Output format:" else "Proceed with API call"
        result = MagicMock()
        result.ask.return_value = answer
        return result

    monkeypatch.setattr(imgedit.questionary, "select", select)
    imgedit.main()
    assert len(saved) == len(ran) == 1
    assert saved[0].extras == {"output_format": fmt}
    if fmt == "gif":
        assert saved[0].res_key == "auto"
        assert saved[0].width is None
    if fmt in ("svg", "gif"):
        assert "Total Cost: depends" in capsys.readouterr().out


def test_back_from_gif_restores_still_geometry(monkeypatch):
    import imgedit

    saved = []

    class Provider(OpenAIProvider):
        def run(self, req):
            pass

    monkeypatch.setattr("sys.argv", ["imgedit.py", "--free", "--no-preview"])
    monkeypatch.setattr(imgedit, "PROVIDER_MAP", {"OpenAI": Provider})
    monkeypatch.setattr(imgedit, "step_provider", lambda: "OpenAI")
    monkeypatch.setattr(imgedit, "step_model", lambda p: "gpt-image-2.6-flare")
    monkeypatch.setattr(
        imgedit, "step_resolution", lambda *a: ("1:1", "1024x1024", 1024, 1024)
    )
    monkeypatch.setattr(imgedit, "step_quality", lambda *a: ("low", 0.0059))
    prompts = iter([(imgedit.BACK_OPTION, None), ("A logo", None)])
    monkeypatch.setattr(imgedit, "step_prompt", lambda *a: next(prompts))
    monkeypatch.setattr(imgedit, "save_last_generation", lambda p, r: saved.append(r))
    formats = iter(["gif", "png"])

    def select(message, **kwargs):
        result = MagicMock()
        result.ask.return_value = (
            next(formats) if message == "Output format:" else "Proceed with API call"
        )
        return result

    monkeypatch.setattr(imgedit.questionary, "select", select)
    imgedit.main()
    assert (saved[0].res_key, saved[0].width, saved[0].height) == (
        "1024x1024",
        1024,
        1024,
    )
