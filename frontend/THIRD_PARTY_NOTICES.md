# Third-party notices

## shadcn/ui — Radix Nova

The following components are vendored from the official shadcn/ui Radix Nova registry.

- `src/components/ui/message.tsx`: https://ui.shadcn.com/r/styles/radix-nova/message.json
- `src/components/ui/trace.tsx`: https://ui.shadcn.com/r/styles/radix-nova/trace.json
- `src/components/ui/collapsible.tsx`: https://ui.shadcn.com/r/styles/radix-nova/collapsible.json
- `src/components/ui/button.tsx`: https://ui.shadcn.com/r/styles/radix-nova/button.json
- `src/components/ui/badge.tsx`: https://ui.shadcn.com/r/styles/radix-nova/badge.json
- `src/components/ui/bubble.tsx`: https://ui.shadcn.com/r/styles/radix-nova/bubble.json
- `src/components/ui/tabs.tsx`: https://ui.shadcn.com/r/styles/radix-nova/tabs.json

License source: https://raw.githubusercontent.com/shadcn-ui/ui/main/LICENSE.md

### Local changes

- Replaced `cn` imports with `@/lib/utils` in message, trace, button, badge, bubble, and tabs.
- Replaced the trace registry collapsible import with `@/components/ui/collapsible`.
- Replaced the `radix-ui` umbrella imports with namespace imports from `@radix-ui/react-slot` in button, badge, and bubble; `@radix-ui/react-collapsible` in collapsible; and `@radix-ui/react-tabs` in tabs.
- Replaced trace's registry-only `IconPlaceholder` import and multi-library icon placeholder with `ChevronRightIcon` from `lucide-react`.
- All other source, styles, behavior, and exports are preserved unchanged.

### License

MIT License

Copyright (c) 2023 shadcn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
