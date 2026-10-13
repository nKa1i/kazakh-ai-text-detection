"""Extract high-resolution visual figures from research progress presentation for the thesis proposal."""

import os
import zipfile


def extract_figures():
    """Extract 7 curated PNG figures from the PPTX archive into proposal/figures/."""
    pptx_path = r"C:\Users\Roza\Desktop\AnekeshD_Progress.pptx"
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "proposal", "figures")
    )
    os.makedirs(out_dir, exist_ok=True)

    mapping = {
        "fig1_fst_fragmentation.png": "ppt/media/image14.png",
        "fig2_overall_framework.png": "ppt/media/image16.png",
        "fig3_dual_stream_fusion.png": "ppt/media/image17.png",
        "fig4_chunk_pooling.png": "ppt/media/image18.png",
        "fig5_fact_checking_trust_matrix.png": "ppt/media/image19.png",
        "fig6_trust_matrix_validation.png": "ppt/media/image24.png",
        "fig7_gradio_prototype.png": "ppt/media/image26.png",
    }

    if not os.path.exists(pptx_path):
        raise FileNotFoundError(f"Source presentation not found at: {pptx_path}")

    with zipfile.ZipFile(pptx_path, "r") as archive:
        for dest_name, src_path in mapping.items():
            dest_file = os.path.join(out_dir, dest_name)
            data = archive.read(src_path)
            with open(dest_file, "wb") as f:
                f.write(data)
            print(f"Extracted {dest_name} ({len(data)} bytes) from {src_path}")


if __name__ == "__main__":
    extract_figures()
