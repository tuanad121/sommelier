# Segmentation Eval Dataset

- Rows: **468**
- Annotated rows (`note` or `correct_script`): **255**

## By video type

- `casual_talkshow`: **80** rows
- `livestream_sales`: **45** rows
- `news_formal`: **14** rows
- `podcast_interview`: **305** rows
- `street_noisy`: **24** rows

## By segmentation relevance

- `high`: **123** rows
- `low`: **258** rows
- `possible`: **87** rows

## Notes

- `segmentation_relevance=high` means note text directly suggests a boundary or segmentation problem.
- `segmentation_relevance=possible` means the row looks risky because it is very short, very long, or a filler/backchannel segment.
- `segmentation_relevance=low` means the row is preserved for context, but there is no direct segmentation signal yet.
