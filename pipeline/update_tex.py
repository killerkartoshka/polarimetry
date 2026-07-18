import os
import glob
from pathlib import Path

def update_tex():
    tex_path = Path("pipeline_report.tex")
    with open(tex_path, "r") as f:
        content = f.read()
        
    # We want to replace the old single gallery figure with a new section
    # Let's locate the marker. The easiest way is to find "Aperture Verification and Star Gallery"
    # and replace the content until "\newpage"
    
    start_str = r"\subsection{Aperture Verification and Star Gallery}"
    end_str = r"\section{Cross-Cycle Tracking and Combination}"
    
    idx_start = content.find(start_str)
    idx_end = content.find(end_str)
    
    if idx_start == -1 or idx_end == -1:
        print("Could not find insertion points!")
        return
        
    # Generate the new LaTeX block for the 14 stars
    # We will use \begin{figure}[H] from float package so they don't jump around.
    
    new_tex = start_str + "\n"
    new_tex += r"To visually verify that our aperture placements align correctly with the physical manifestations of both the ordinary PSFs and the extraordinary streaks, we generated a comprehensive visual catalog of the 14 matched targets overlapping with the Kanata Observatory reference catalog." + "\n\n"
    new_tex += r"Each target is displayed in a triple-panel format. The left panel shows the full dual-beam context, highlighting where the ordinary and extraordinary beams physically land on the CCD. The middle and right panels show the zoomed-in, master-subtracted ordinary and extraordinary beams respectively, alongside their designated circular ($r=12\text{ px}$) and elliptical ($a=50, b=12\text{ px}$) photometry apertures." + "\n\n"
    
    for i in range(1, 15):
        new_tex += r"\begin{figure}[htbp]" + "\n"
        new_tex += r"\centering" + "\n"
        new_tex += rf"\includegraphics[width=\textwidth]{{figures/matched_star_{i:02d}.png}}" + "\n"
        new_tex += rf"\caption{{Aperture geometry and polarization comparison for Matched Star \#{i}.}}" + "\n"
        new_tex += r"\end{figure}" + "\n\n"
        if i % 3 == 0:
            new_tex += r"\clearpage" + "\n\n"
            
    new_tex += r"\clearpage" + "\n\n"
    
    final_content = content[:idx_start] + new_tex + content[idx_end:]
    
    # We need to make sure \usepackage{float} is in the header
    if r"\usepackage{float}" not in final_content:
        pkg_idx = final_content.find(r"\usepackage{cite}")
        final_content = final_content[:pkg_idx] + r"\usepackage{cite}" + "\n" + r"\usepackage{float}" + final_content[pkg_idx+18:]
        
    with open(tex_path, "w") as f:
        f.write(final_content)
        
    print("Updated pipeline_report.tex with 14 matched stars gallery.")

if __name__ == "__main__":
    update_tex()
