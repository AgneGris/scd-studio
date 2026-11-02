"""Scroll area that lets tall pages and sidebars fit short screens."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QWidget


class VerticalScrollArea(QScrollArea):
    """Vertical-only scroll area that is never narrower than its content."""

    def __init__(self, content: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(content)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        content = self.widget()
        if content is None:
            return hint
        width = (
            content.minimumSizeHint().width()
            + 2 * self.frameWidth()
            + self.verticalScrollBar().sizeHint().width()
        )
        return QSize(max(hint.width(), width), hint.height())

    def eventFilter(self, obj, event):
        if obj is self.widget() and event.type() == QEvent.Type.LayoutRequest:
            self.updateGeometry()
        return super().eventFilter(obj, event)
