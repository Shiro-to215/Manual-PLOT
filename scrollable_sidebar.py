"""Scrollable settings panel with separate scrolling for its joint list."""
import tkinter as tk
from tkinter import ttk


class ScrollableSidebar(ttk.Frame):
    def __init__(self, parent, width=320):
        super().__init__(parent, width=width)
        self.pack_propagate(False)
        background = ttk.Style(self).lookup('TFrame', 'background')
        self.canvas = tk.Canvas(self, width=width - 20, highlightthickness=0,
                                background=background, yscrollincrement=24)
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.content = ttk.Frame(self.canvas, padding=12)
        self.window = self.canvas.create_window(0, 0, window=self.content, anchor='nw')
        self.content.bind('<Configure>', self._update_region)
        self.canvas.bind('<Configure>', self._resize)
        self._wheel_tag = f'SidebarWheel:{self}'
        for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.bind_class(self._wheel_tag, sequence, self._wheel)

    def _resize(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)
        self._update_region()

    def _update_region(self, event=None):
        height = max(self.content.winfo_reqheight(), self.canvas.winfo_height())
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), height))
        if self.content.winfo_reqheight() <= self.canvas.winfo_height():
            self.canvas.yview_moveto(0)

    def bind_mousewheel(self):
        """Run after building the controls; route events before widget class bindings."""
        def visit(widget):
            tags = widget.bindtags()
            if self._wheel_tag not in tags:
                widget.bindtags((tags[0], self._wheel_tag, *tags[1:]))
            for child in widget.winfo_children():
                visit(child)
        visit(self)

    def _wheel(self, event):
        if getattr(event, 'num', None) in (4, 5):
            steps = -1 if event.num == 4 else 1
        else:
            delta = event.delta
            if not delta:
                return 'break'
            steps = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        # Scroll the joint list itself while it has content in that direction.
        if isinstance(event.widget, tk.Listbox):
            first, last = event.widget.yview()
            if (steps < 0 and first > 0) or (steps > 0 and last < 1):
                return None
        self.canvas.yview_scroll(steps, 'units')
        return 'break'
