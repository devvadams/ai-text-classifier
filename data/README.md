# SMS Spam Collection dataset

## Attribution and license

Tiago Almeida and José María Gómez Hidalgo. **SMS Spam Collection** (2012).
UCI Machine Learning Repository. DOI: [10.24432/C5CC84](https://doi.org/10.24432/C5CC84).

- Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Official archive: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
  Attribution and an indication of changes are required when redistributing it.
  The repository's MIT license does **not** replace the dataset's license.

Related paper: Almeida, T. A., Gómez Hidalgo, J. M., and Yamakami, A. (2011).
*Contributions to the study of SMS spam filtering: new collection and results.*
Proceedings of the ACM Symposium on Document Engineering.
DOI: [10.1145/2034691.2034742](https://doi.org/10.1145/2034691.2034742).

## Included file and conversion

`sms_spam.csv` is a UTF-8 CSV with the header `label,text`. It contains the real
SMS Spam Collection, not synthetic examples:

| Label | Meaning | Records |
| --- | --- | ---: |
| `ham` | Legitimate SMS | 4,827 |
| `spam` | Spam SMS | 747 |
| **Total** | | **5,574** |

The source used for this conversion is the collection's TSV mirror in
[`justmarkham/pycon-2016-tutorial`](https://github.com/justmarkham/pycon-2016-tutorial/blob/61c51b26e73ecfc6aa774aea3673c3b122804343/data/sms.tsv),
pinned to commit `61c51b26e73ecfc6aa774aea3673c3b122804343`.

Changes made: split each source line at its **first tab**, retain the `ham`/`spam`
labels and message text, add a `label,text` header, and write CSV quoting with
Python's standard-library `csv.writer` and LF record endings. No records were
sampled, fabricated, or deduplicated in the included file. CSV quoting preserves
commas and double quotes inside messages.

SHA-256 checksums:

```text
Source TSV:   7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d
sms_spam.csv: f6605cf7f69b9cb2f9518355fe104be11b6c6ac0c01fb0200caa6821fd7f945d
```

The training loader strips leading/trailing whitespace and removes identical
messages before splitting; this leaves 5,160 unique messages. It rejects
conflicting labels instead of silently resolving them. The CSV stays unchanged.

This historical corpus includes UK/Singapore SMS conventions and is not
representative of every language, region, or present-day spam campaign. Treat
phone numbers and URLs inside messages as dataset content, not instructions to
contact or visit them.
