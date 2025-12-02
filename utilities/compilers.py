import os, glob


def find_cl_path():
    """
    Locate MSVC cl.exe on Windows for torch.compile compatibility.
    Returns the path to the newest cl.exe if found, else None.
    """
    roots = [
        r"C:\\Program Files (x86)\\Microsoft Visual Studio",
        r"C:\\Program Files\\Microsoft Visual Studio",
    ]
    found_paths = []
    for root in roots:
        if not os.path.exists(root):
            continue
        pattern = os.path.join(root, "*", "*", "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe")
        found_paths.extend(glob.glob(pattern))
    if not found_paths:
        return None
    found_paths.sort(reverse=True)
    return found_paths[0]
