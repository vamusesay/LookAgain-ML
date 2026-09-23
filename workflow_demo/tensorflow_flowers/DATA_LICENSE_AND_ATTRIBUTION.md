# TensorFlow Flowers data licence and attribution

The preparation script obtains the official TensorFlow Flowers archive from
<https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz>.
The archive's `flower_photos/LICENSE.txt` states that its images use
Creative Commons Attribution 2.0 and supplies per-image photographer credits
and original Flickr links. Retain that file and follow its attribution terms
when using images. The applicable licence is
<https://creativecommons.org/licenses/by/2.0/>.

No images or dataset archive are bundled here. The preparation script verifies
the archive size (228813984 bytes) and SHA-256
`4C54ACE7911AAFFE13A365C34F650E71DD5BF1BE0A58B464E5A7183E3E595D9C`
and requires the extracted `LICENSE.txt`. Author credits belong to the dataset's
third-party photographers; they are not project authorship claims.

LookAgain-ML's MIT code licence does not license third-party data or encoder
weights. Checkpoint metadata and upstream terms are recorded separately in
`environment/encoder_checkpoints.csv` at the repository root. Restricted paper
datasets must be obtained independently under their applicable access terms.
