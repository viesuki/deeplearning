---
title: Settings
enableToc: True
---

# Frontmatters

| Field               | Purpose                                                                                 | Aliases                               | Default                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `title`             | Page title; if omitted, the filename is used                                            | —                                     | Filename is used as the title. ([Quartz][1])                                                             |
| `description`       | Description used for link previews/SEO                                                  | —                                     | None (may be filled by a social card/default template). ([Quartz][1])                                    |
| `permalink`         | Permanent link (stable even if the file path changes)                                   | —                                     | Generated from the file path. ([Quartz][1])                                                              |
| `tags`              | Tags (supports hierarchies like `nlp/qa`)                                               | `tag`                                 | Empty; Quartz generates tag pages and indexes from what you provide. ([Quartz][1])                       |
| `aliases`           | Page aliases (for internal links/search)                                                | `alias`                               | Empty. ([Quartz][1])                                                                                     |
| `cssclasses`        | Extra CSS class names attached to the page container                                    | `cssclass`                            | Empty. ([Quartz][1])                                                                                     |
| `lang`              | Page language (affects `html lang` and some typographic details)                        | —                                     | Inherits site setting. ([Quartz][2])                                                                     |
| `draft`             | Whether **not** to publish (private/draft)                                              | —                                     | `false`; the **RemoveDrafts** filter omits pages with `draft: true`. ([Quartz][3])                       |
| `publish`           | Publish only pages with `publish: true` (effective when **ExplicitPublish** is enabled) | —                                     | Only effective if you switch to ExplicitPublish. ([Quartz][3])                                           |
| `comments`          | Show the comments component (e.g., Giscus)                                              | —                                     | **Enabled by default**; set `false` on a page to disable. ([Quartz][4])                                  |
| `enableToc`         | Show the right-hand TOC                                                                 | —                                     | **Enabled by default** (when there are ≥2 headings); set `false` to disable. ([Quartz][5])               |
| `socialDescription` | Social share card description                                                           | — (falls back to `description`)       | None; overridden if **Custom OG Images** is enabled. ([Quartz][6])                                       |
| `socialImage`       | Social share card image                                                                 | `image`, `cover`                      | If generation is enabled, priority is “frontmatter > generated > default image.” ([Anson | Git][7])     |
| `created`           | Created date                                                                            | `date`                                | Determined by **CreatedModifiedDate** plugin by priority (frontmatter → git → filesystem). ([Quartz][8]) |
| `modified`          | Modified date                                                                           | `lastmod`, `updated`, `last-modified` | Same; if omitted, may be taken from Git/filesystem. ([Quartz][8])                                        |
| `published`         | Publication date                                                                        | `publishDate`, `date`                 | Same; **recommend using `publishDate`** to avoid ambiguity with `date`. ([Quartz][2])                    |


[1]: https://quartz.jzhao.xyz/authoring-content "Authoring Content"
[2]: https://quartz.jzhao.xyz/plugins/Frontmatter "Frontmatter"
[3]: https://quartz.jzhao.xyz/features/private-pages?utm_source=chatgpt.com "Private Pages"
[4]: https://quartz.jzhao.xyz/features/comments?utm_source=chatgpt.com "Comments"
[5]: https://quartz.jzhao.xyz/features/table-of-contents?utm_source=chatgpt.com "Table of Contents"
[6]: https://quartz.jzhao.xyz/plugins/CustomOgImages?utm_source=chatgpt.com "Custom OG Images"
[7]: https://git.ansonbiggs.com/Anson/brain-quartz/src/commit/7be47742a6dc86f22d148ca9d304f7a9eea318cf/docs/features/social%20images.md?utm_source=chatgpt.com "brain-quartz - Anson | Git"
[8]: https://quartz.jzhao.xyz/plugins/CreatedModifiedDate "CreatedModifiedDate"


