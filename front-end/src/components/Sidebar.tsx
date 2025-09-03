
import React, { useState, useRef, useEffect } from "react";
import { FaFolderOpen, FaCog, FaUserCircle } from "react-icons/fa";
import "./Sidebar.css";

interface SidebarProps {
  active: string;
  onNavigate: (page: string) => void;
  onAddPage?: () => void;
  onDeletePage?: (key: string) => void;
  onRenamePage?: (key: string) => void;
  onReorder?: (fromKey: string, toIndex: number) => void;
  pages: { key: string; label: string }[];
}

// pages are provided by the parent (App) so the sidebar reflects dynamic additions

const Sidebar: React.FC<SidebarProps> = ({ active, onNavigate, onAddPage, onDeletePage, onRenamePage, onReorder, pages }) => {
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const sidebarRef = useRef<HTMLDivElement | null>(null);
  const handleRef = useRef<HTMLDivElement | null>(null);

  // Sidebar width state is stored as a CSS variable on the root so it's easy to read from CSS.
  useEffect(() => {
    const saved = localStorage.getItem("sidebar-width");
    if (saved) {
      document.documentElement.style.setProperty("--sidebar-width", saved);
    }
  }, []);

  useEffect(() => {
    const handle = handleRef.current;
    if (!handle) return;

    let startX = 0;
    let startWidth = 0;

    const onPointerDown = (e: PointerEvent) => {
      startX = e.clientX;
      const computed = getComputedStyle(document.documentElement).getPropertyValue("--sidebar-width") || "260px";
      startWidth = parseInt(computed, 10) || 260;
      (e.target as Element).setPointerCapture(e.pointerId);
      document.addEventListener("pointermove", onPointerMove);
      document.addEventListener("pointerup", onPointerUp);
    };

    const onPointerMove = (e: PointerEvent) => {
      const dx = e.clientX - startX;
      let newWidth = startWidth + dx;
      const min = 200; // minimum width in px
      const max = 420; // maximum width
      if (newWidth < min) newWidth = min;
      if (newWidth > max) newWidth = max;
      const value = `${newWidth}px`;
      document.documentElement.style.setProperty("--sidebar-width", value);
      localStorage.setItem("sidebar-width", value);
    };

    const onPointerUp = (e: PointerEvent) => {
      try { (e.target as Element).releasePointerCapture(e.pointerId); } catch {}
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", onPointerUp);
    };

    handle.addEventListener("pointerdown", onPointerDown);
    return () => handle.removeEventListener("pointerdown", onPointerDown);
  }, []);

  // Close open item menus when clicking outside the sidebar
  useEffect(() => {
    const handleDocClick = (e: MouseEvent) => {
      if (sidebarRef.current && !sidebarRef.current.contains(e.target as Node)) {
        setOpenMenu(null);
      }
    };

    document.addEventListener("mousedown", handleDocClick);
    return () => document.removeEventListener("mousedown", handleDocClick);
  }, []);

  return (
    <aside className="sidebar" ref={sidebarRef}>
      <div className="sidebar-resize-handle" ref={handleRef} title="Resize sidebar" />
      <div className="sidebar-header">
        <span className="logo">Blackline Forensics <span className="for-figma">AI</span></span>
      </div>
      <div className="sidebar-profile">
        <div className="profile-left">
          <FaUserCircle size={36} />
        </div>
        <div className="profile-actions">
          <FaCog className="sidebar-icon" />
        </div>
      </div>
      {/* render NEW ANALYSIS as its own centered box so it can resize with the sidebar */}
      <div className="home-container">
        <div
          className="home-replace"
          onClick={() => { onAddPage && onAddPage(); }}
          role="button"
          tabIndex={0}
        >
          <button
            className="home-add-icon"
            title="New analysis"
            onClick={(e) => { e.stopPropagation(); onAddPage && onAddPage(); }}
          >
            +
          </button>
          <span className="new-bubble">NEW ANALYSIS</span>
        </div>
      </div>
  <nav className="sidebar-nav">
        {pages.filter(p => p.key !== 'dashboard').map((item, idx) => (
          <div
            key={item.key}
            className={`sidebar-item${(active === item.key || openMenu === item.key) ? " active" : ""}`}
            draggable={true}
            onDragStart={(e) => {
              e.dataTransfer?.setData('text/plain', item.key);
              e.dataTransfer!.effectAllowed = 'move';
              // add dragging class for visual
              (e.currentTarget as HTMLElement).classList.add('dragging');
            }}
            onDragEnd={(e) => {
              (e.currentTarget as HTMLElement).classList.remove('dragging');
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.dataTransfer!.dropEffect = 'move';
              const el = e.currentTarget as HTMLElement;
              // determine whether pointer is in top or bottom half
              const rect = el.getBoundingClientRect();
              const mid = rect.top + rect.height / 2;
              el.classList.remove('drag-over', 'drop-above', 'drop-below');
              if (e.clientY < mid) {
                el.classList.add('drop-above');
              } else {
                el.classList.add('drop-below');
              }
            }}
            onDragLeave={(e) => {
              const el = e.currentTarget as HTMLElement;
              el.classList.remove('drag-over', 'drop-above', 'drop-below');
            }}
            onDrop={(e) => {
              e.preventDefault();
              const el = e.currentTarget as HTMLElement;
              const fromKey = e.dataTransfer?.getData('text/plain');
              if (!fromKey) return;
              // If hovering on top half -> insert before this item (idx), else insert after (idx + 1)
              const insertBefore = el.classList.contains('drop-above');
              const targetIndex = insertBefore ? idx : idx + 1;
              el.classList.remove('drag-over', 'drop-above', 'drop-below');
              onReorder && onReorder(fromKey, targetIndex);
            }}
          >
            <div
              className="sidebar-link"
              onClick={() => {
                if (item.key === 'dashboard') {
                  onAddPage && onAddPage();
                } else {
                  onNavigate(item.key);
                }
              }}
              role="button"
              tabIndex={0}
            >
              <div className="sidebar-link-left">
                {item.key === 'dashboard' ? (
                  <div className="home-replace">
                    <button
                      className="home-add-icon"
                      title="New analysis"
                      onClick={(e) => { e.stopPropagation(); onAddPage && onAddPage(); }}
                    >
                      +
                    </button>
                    <span className="new-bubble">NEW ANALYSIS</span>
                  </div>
                ) : (
                  <>
                    <FaFolderOpen /> <span className="sidebar-label">{item.label}</span>
                  </>
                )}
              </div>
              {/* action: three dots - visible on hover or when active; don't show for dashboard */}
              {item.key !== 'dashboard' && (
                <div className="sidebar-actions-wrap">
                  <button
                    className="item-actions"
                    title="More"
                    onClick={(e) => { e.stopPropagation(); setOpenMenu(openMenu === item.key ? null : item.key); }}
                  >
                    &#8230;
                  </button>

                  {openMenu === item.key && (
                    <div className="item-menu" onClick={(e) => e.stopPropagation()}>
                      <button className="item-menu-btn" onClick={() => { onRenamePage && onRenamePage(item.key); setOpenMenu(null); }}>Rename</button>
                      <button className="item-menu-btn danger" onClick={() => { onDeletePage && onDeletePage(item.key); setOpenMenu(null); }}>Delete</button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </nav>
  <input className="sidebar-search" placeholder="Search for..." />
    </aside>
  );
};

export default Sidebar;
