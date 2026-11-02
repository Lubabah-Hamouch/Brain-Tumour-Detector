/**
 * Plain-language definitions for the domain terms the screens use, for a visitor without a
 * medical-imaging background. Shown on demand by the Term component, never as page text.
 */

export const GLOSSARY = {
  nifti: {
    term: "NIfTI (.nii, .nii.gz)",
    text: "The standard file format for 3D medical scans such as MRI. One file holds a whole scan volume.",
  },
  sequences: {
    term: "FLAIR, T1, T1ce, T2",
    text:
      "Four kinds of MRI scan of the same head, each showing tissue differently: FLAIR highlights swelling, T1 shows " +
      "anatomy, T1ce is T1 after a contrast injection that lights up active tumour, and T2 shows fluid. The model " +
      "reads all four together.",
  },
  flair: {
    term: "FLAIR",
    text:
      "Fluid-attenuated inversion recovery: a T2-weighted scan with the fluid signal suppressed, so swelling (edema) " +
      "around a tumour stands out bright. Best for seeing the whole tumour.",
  },
  t1: {
    term: "T1",
    text: "The plain anatomical scan: grey and white matter are clearly separated, and fluid is dark. No contrast agent.",
  },
  t1ce: {
    term: "T1ce",
    text:
      "T1 after a contrast agent (gadolinium) is injected. Actively growing tumour takes up the agent and lights up, " +
      "which is how the enhancing tumour is found.",
  },
  t2: {
    term: "T2",
    text: "A scan where fluid is bright: it shows swelling and the fluid-filled spaces of the brain, and helps outline the tumour core.",
  },
  coregistered: {
    term: "Co-registered",
    text: "Aligned to one shared grid, so the same voxel in each scan is the same point in the head.",
  },
  skullStripped: {
    term: "Skull-stripped",
    text: "Skull and scalp removed so only the brain remains: the standard preparation for this kind of data.",
  },
  brats: {
    term: "BraTS",
    text: "The Brain Tumor Segmentation challenge dataset: MRI of 1,251 glioma patients, each with tumour regions outlined by experts.",
  },
  dice: {
    term: "Dice score",
    text: "How well a predicted region overlaps the region outlined by experts, from 0 (no overlap) to 1 (identical).",
  },
  regions: {
    term: "WT, TC, ET",
    text:
      "The three tumour regions scored in BraTS. Whole tumour (WT): everything abnormal, including the swelling " +
      "around the tumour. Tumour core (TC): the tumour itself, without the swelling. Enhancing tumour (ET): its " +
      "actively growing part.",
  },
  wt: {
    term: "Whole tumour (WT)",
    text: "Everything abnormal: the tumour plus the swelling (edema) around it. Labels 1, 2 and 4.",
  },
  tc: {
    term: "Tumour core (TC)",
    text: "The tumour itself without the surrounding swelling: its dead (necrotic) centre plus its actively growing part. Labels 1 and 4.",
  },
  et: {
    term: "Enhancing tumour (ET)",
    text: "The actively growing part of the tumour, which lights up on the contrast-enhanced (T1ce) scan. Label 4.",
  },
  labels: {
    term: "Labels",
    text:
      "What each voxel is predicted to be: 2 edema (swelling), 1 necrotic core (dead tumour tissue), 4 enhancing " +
      "tumour (actively growing). Shown in shades of red.",
  },
  ensemble: {
    term: "5-fold ensemble",
    text:
      "Five copies of the network, each trained on a different 80% of the training patients. Their predictions are " +
      "averaged, which is more reliable than any one of them.",
  },
  agreement: {
    term: "Agreement",
    text:
      "How closely the five models agree on this region (their mean overlap with the averaged result). High agreement " +
      "means a consistent prediction, not necessarily a correct one.",
  },
  meanP: {
    term: "Mean p",
    text: "The models' average confidence (probability) over the voxels inside this region.",
  },
  etCleanup: {
    term: "ET clean-up",
    text: "Tiny isolated specks of predicted enhancing tumour (under 50 voxels) are removed; they are almost always false alarms.",
  },
  slice: {
    term: "2D slice",
    text: "One flat cross-section image taken from a 3D scan, such as a typical MRI picture.",
  },
  classes: {
    term: "The four classes",
    text: "Glioma, meningioma and pituitary tumour, three common kinds of brain tumour, or no tumour.",
  },
  gradcam: {
    term: "Grad-CAM",
    text:
      "A heatmap of the image areas that most influenced the prediction, whatever that prediction is. It is coarse " +
      "(12 × 12 cells) and shows where the model looked, not a precise outline of the tumour. For a “No tumour” " +
      "result, a highlighted region does not indicate a lesion.",
  },
  letterbox: {
    term: "Letterbox",
    text: "The image is scaled to fit 384 × 384 px without stretching, with black bars filling the rest: exactly as in training.",
  },
  knownError: {
    term: "Known error",
    text: "One of the test images the model gets wrong, included so you can see what a failure looks like.",
  },
  heldOut: {
    term: "Held-out test set",
    text: "Data set aside before training and never used for training or tuning, so it measures performance on unseen cases.",
  },
  ci: {
    term: "95% confidence interval",
    text: "The range the true accuracy plausibly lies in, given how many test images there are.",
  },
  macroF1: {
    term: "Macro F1",
    text:
      "A score from 0 to 1 that balances missed cases and false alarms, averaged over the four classes equally so " +
      "weakness on one class is not hidden.",
  },
  stress128: {
    term: "Shrunk to 128 px",
    text:
      "A stress test: every test image is resized to 128 × 128 px. The first model had learned that small files " +
      "usually meant “no tumour”; this checks that the served model judges the image, not its size.",
  },
  crossValidation: {
    term: "Cross-validation",
    text: "The training patients are split into five parts; each model is checked on the part it did not train on.",
  },
  mcnemar: {
    term: "McNemar test",
    text: "Checks whether two models' mistakes on the same images differ by more than chance. A p-value of 1.0 means no evidence of a difference.",
  },
  pairedCi: {
    term: "Paired comparison",
    text:
      "Both versions are scored on the same test patients and the difference is measured patient by patient; the " +
      "95% interval shows how sure that difference is. An interval that includes 0 means no clear difference.",
  },
  mask: {
    term: "Label map (.nii.gz)",
    text: "The full 3D result as a NIfTI file: every voxel marked with its predicted label, aligned with the input scans. Opens in medical viewers such as ITK-SNAP.",
  },
  glb: {
    term: ".glb",
    text: "A standard 3D model file (glTF). Opens in most 3D viewers and tools, including Windows 3D Viewer and Blender.",
  },
} as const;

export type GlossaryKey = keyof typeof GLOSSARY;

/** Glossary entry for one MRI sequence, by its display name (FLAIR, T1, T1ce, T2). */
export const sequenceTerm = (name: string) => name.toLowerCase() as "flair" | "t1" | "t1ce" | "t2";
