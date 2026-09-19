# Clone the OpenAD reference code (MIT, (c) 2023 Toan Nguyen) at a pinned commit.
# We reference OpenAD; we do not vendor it into this repo. Our code in src/ imports
# from the cloned checkout at ./_ref_openad.
$ErrorActionPreference = "Stop"

$Repo = "https://github.com/Fsoft-AIC/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds"
$PinnedCommit = "b082265ed085455a3fdc80b0d8d95dc51351e7b4"
$Dest = "_ref_openad"

if (Test-Path "$Dest/.git") {
    Write-Host "'$Dest' already exists; leaving it untouched."
    exit 0
}

Write-Host "Cloning OpenAD into '$Dest' ..."
git clone $Repo $Dest
git -C $Dest checkout $PinnedCommit
Write-Host "Done. OpenAD checked out at $PinnedCommit."
