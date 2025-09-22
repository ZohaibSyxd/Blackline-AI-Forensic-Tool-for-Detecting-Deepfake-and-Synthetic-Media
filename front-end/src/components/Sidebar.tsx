
import React, { useState, useEffect, useRef } from 'react';
import { DragDropContext, Droppable, Draggable, DropResult } from '@hello-pangea/dnd';
import './Sidebar.css';
import ConfirmDialog from './ConfirmDialog';
import NewAnalysisModal from './NewAnalysisModal';
import ProfilePopup from './ProfilePopup';
import IconPickerModal from './IconPickerModal';
import SettingsModal from './SettingsModal';
import userProfileIcon from '../../assets/icons8-profile-100.png';
import settingsIcon from '../../assets/icons8-setting-100.png';
import { initTheme } from '../theme';

interface SidebarProps {
  active: string;
  onNavigate: (page: string) => void;
  onAddPage?: () => void;
  onDeletePage?: (key: string) => void;
  onBulkDelete?: (keys: string[]) => void;
  onRenamePage?: (key: string, newName?: string) => void;
  onReorder?: (fromKey: string, toIndex: number) => void;
  onChangeIcon?: (key: string, icon?: string) => void;
  pages: { key: string; label: string; icon?: string }[];
}

// pages are provided by the parent (App) so the sidebar reflects dynamic additions

const Sidebar: React.FC<SidebarProps> = ({ active, onNavigate, onAddPage, onDeletePage, onBulkDelete, onRenamePage, onReorder, onChangeIcon, pages }) => {
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState<boolean>(() => localStorage.getItem('sidebar-collapsed') === '1');
  // multi-select state for bulk actions
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const lastClickedRef = useRef<string | null>(null);
  const sidebarRef = useRef<HTMLDivElement | null>(null);
  const handleRef = useRef<HTMLDivElement | null>(null);
  // inline rename state
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState<string>("");
  const editingInputRef = useRef<HTMLInputElement | null>(null);
  // confirm dialog state
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmMessage, setConfirmMessage] = useState('');
  const confirmActionRef = useRef<(() => void) | null>(null);
  // search state
  const [searchQuery, setSearchQuery] = useState<string>('');
  // icon picker state
  const [iconPickerOpen, setIconPickerOpen] = useState(false);
  const [iconPickerKey, setIconPickerKey] = useState<string | null>(null);
  // settings modal state
  const [settingsOpen, setSettingsOpen] = useState(false);

  const askConfirm = (message: string, onConfirm: () => void) => {
    setConfirmMessage(message);
    confirmActionRef.current = onConfirm;
    setConfirmOpen(true);
  };

  // Sidebar width state is stored as a CSS variable on the root so it's easy to read from CSS.
  useEffect(() => {
    const saved = localStorage.getItem("sidebar-width");
    if (saved) {
      document.documentElement.style.setProperty("--sidebar-width", saved);
    }
    if (collapsed) {
      document.documentElement.style.setProperty('--sidebar-width', '72px');
    }
    // Initialize theme (light/dark/system)
    let dispose: undefined | (() => void);
    try { const res = initTheme(); dispose = res.dispose; } catch {}
    return () => { if (dispose) dispose(); };
  }, []);

  useEffect(() => {
    const handle = handleRef.current;
    if (!handle) return;
    if (collapsed) return; // resizing disabled when collapsed; use button to expand/collapse

    let startX = 0;
    let startWidth = 0;
    let currentWidth = 0;

    const onPointerDown = (e: PointerEvent) => {
      // Avoid selecting text/content when starting a resize drag
      e.preventDefault();
      startX = e.clientX;
      const computed = getComputedStyle(document.documentElement).getPropertyValue("--sidebar-width") || "260px";
      startWidth = parseInt(computed, 10) || 260;
      currentWidth = startWidth;
      (e.target as Element).setPointerCapture(e.pointerId);
      // disable transitions during manual resize for responsiveness
      if (sidebarRef.current) sidebarRef.current.classList.add('resizing');
      // disable global selection while dragging
      document.body.classList.add('sidebar-resizing-global');
      document.addEventListener("pointermove", onPointerMove);
      document.addEventListener("pointerup", onPointerUp);
    };

    const onPointerMove = (e: PointerEvent) => {
      // Avoid accidental selections during drag
      e.preventDefault();
      const dx = e.clientX - startX;
      const rawWidth = startWidth + dx;
  const min = 280; // respect expanded min width during drag
      const max = 420;
      let newWidth = rawWidth;
      if (newWidth < min) newWidth = min;
      if (newWidth > max) newWidth = max;
      currentWidth = newWidth;
      const value = `${newWidth}px`;
      document.documentElement.style.setProperty("--sidebar-width", value);
        // Avoid writing to localStorage on every frame to keep drag smooth
    };

    const onPointerUp = (e: PointerEvent) => {
      try { (e.target as Element).releasePointerCapture(e.pointerId); } catch {}
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", onPointerUp);
      // re-enable transitions after resize completes
      if (sidebarRef.current) sidebarRef.current.classList.remove('resizing');
      document.body.classList.remove('sidebar-resizing-global');
      // Clamp and persist the final expanded width, and apply it immediately
  const clamped = Math.max(280, Math.min(420, currentWidth));
      const value = `${clamped}px`;
      document.documentElement.style.setProperty("--sidebar-width", value);
      localStorage.setItem("sidebar-width", value);
    };

    handle.addEventListener("pointerdown", onPointerDown);
    return () => handle.removeEventListener("pointerdown", onPointerDown);
  }, [collapsed]);

  // persist collapsed state changes and adjust width
  useEffect(() => {
    localStorage.setItem('sidebar-collapsed', collapsed ? '1' : '0');
    if (collapsed) {
      document.documentElement.style.setProperty('--sidebar-width', '72px');
    } else {
      const saved = localStorage.getItem('sidebar-width') || '300px';
      document.documentElement.style.setProperty('--sidebar-width', saved);
    }
  }, [collapsed]);

  // Close open item menus when clicking outside the sidebar
  useEffect(() => {
    const handleDocClick = (e: MouseEvent) => {
      const target = e.target as Element;
      const inMenu = target.closest('.item-menu');
      const inActionsBtn = target.closest('.item-actions');
      // If click is not on the menu or the actions button, close the open menu
      if (!inMenu && !inActionsBtn) {
        setOpenMenu(null);
      }

      // Preserve previous behavior: if clicking completely outside the sidebar,
      // commit any in-progress rename and exit edit mode.
      if (sidebarRef.current && !sidebarRef.current.contains(target)) {
        if (editingKey && editingValue.trim()) {
          onRenamePage && onRenamePage(editingKey, editingValue.trim());
        }
        setEditingKey(null);
      }
    };

    document.addEventListener("mousedown", handleDocClick);
    return () => document.removeEventListener("mousedown", handleDocClick);
  }, []);

  // focus input when entering edit mode
  useEffect(() => {
    if (editingKey && editingInputRef.current) {
      editingInputRef.current.focus();
      editingInputRef.current.select();
    }
  }, [editingKey]);

  const startInlineRename = (itemKey: string, currentLabel: string) => {
    if (itemKey === 'dashboard') return;
    setEditingKey(itemKey);
    setEditingValue(currentLabel);
    setOpenMenu(null);
  };

  const commitInlineRename = () => {
    if (!editingKey) return;
    const newName = editingValue.trim();
    if (newName) {
      onRenamePage && onRenamePage(editingKey, newName);
    }
    setEditingKey(null);
  };

  const cancelInlineRename = () => {
    setEditingKey(null);
  };

  // compute list to display (exclude dashboard, and filter by search)
  const q = searchQuery.trim().toLowerCase();
  const allNonDashboard = pages.filter(p => p.key !== 'dashboard');
  // tokenizes a string into alphanumeric "words" (letters/digits), splitting on spaces and punctuation
  const getWords = (s: string) => (s.toLowerCase().match(/[a-z0-9]+/g) || []);
  const matchesWordPrefix = (s: string, query: string) => {
    if (!query) return true;
    const words = getWords(s);
    return words.some(w => w.startsWith(query));
  };
  const displayedPages = q
    ? allNonDashboard.filter(p => matchesWordPrefix(p.label, q) || matchesWordPrefix(p.key, q))
    : allNonDashboard;

  return (
    <aside className={`sidebar${collapsed ? ' collapsed' : ''}`} ref={sidebarRef}>
      <div className="sidebar-resize-handle" ref={handleRef} title="Resize sidebar" />
      <div className="sidebar-header">
        {!collapsed && (
          <span className="logo">Blackline Forensics <span className="for-figma">AI</span></span>
        )}
        <button
          className="collapse-toggle"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={collapsed ? 'Expand' : 'Collapse'}
          onClick={() => setCollapsed(v => !v)}
        >
          {collapsed ? '›' : '‹'}
        </button>
      </div>
  {/* top area intentionally kept minimal (logo above). Profile and settings moved to the footer below the search */}
      {/* render NEW ANALYSIS button depending on collapsed state */}
      {!collapsed ? (
        <div className="home-container">
          <button className="home-replace-button" onClick={() => { onAddPage && onAddPage(); }}>
            <span className="home-add-icon" aria-hidden>+</span>
            <span className="new-bubble">NEW ANALYSIS</span>
          </button>
        </div>
      ) : (
        <div className="home-compact">
          <button className="home-compact-btn" onClick={() => { onAddPage && onAddPage(); }} title="New analysis" aria-label="New analysis">+</button>
        </div>
      )}

      <nav className="sidebar-nav">
        {collapsed ? (
          <div className="collapsed-icon-list">
            {displayedPages.map(item => (
              <button
                key={item.key}
                className={`collapsed-icon-btn${active === item.key ? ' active' : ''}`}
                title={item.label}
                aria-label={item.label}
                onClick={() => onNavigate(item.key)}
              >
                <span className="sidebar-item-icon" aria-hidden>{item.icon || '📁'}</span>
              </button>
            ))}
          </div>
        ) : (
        q === '' ? (
          <DragDropContext onDragEnd={(result: DropResult) => {
            const { source, destination } = result;
            if (!destination) return;
            // since search is empty, displayedPages === allNonDashboard
            const fromKey = displayedPages[source.index].key;
            const toIndex = destination.index;
            onReorder && onReorder(fromKey, toIndex);
          }}>
            <Droppable droppableId="sidebar-pages">
              {(provided) => (
                <div ref={provided.innerRef} {...provided.droppableProps}>
                {/* bulk action bar shown when there are selected items */}
                {selectedKeys.length > 0 && (
                  <div className="sidebar-bulk-bar">
                    <div className="bulk-info">{selectedKeys.length} selected</div>
                    <div className="bulk-actions">
                      <button className="item-menu-btn" onClick={() => { setSelectedKeys([]); }}>Clear</button>
                      <button
                        className="item-menu-btn danger"
                        onClick={() => {
                          askConfirm(`Delete ${selectedKeys.length} page${selectedKeys.length === 1 ? '' : 's'}? This action cannot be undone.`, () => {
                            if (onBulkDelete) {
                              onBulkDelete(selectedKeys);
                            } else if (onDeletePage) {
                              selectedKeys.forEach(k => onDeletePage(k));
                            }
                            setSelectedKeys([]);
                            setConfirmOpen(false);
                          });
                        }}
                      >Delete</button>
                    </div>
                  </div>
                )}
                {displayedPages.map((item, idx) => {
                  const isSelected = selectedKeys.includes(item.key);
                  return (
                    <Draggable key={item.key} draggableId={item.key} index={idx}>
                      {(providedDraggable, snapshot) => {
                        return (
                          <div
                            ref={providedDraggable.innerRef}
                            {...providedDraggable.draggableProps}
                            className={`sidebar-item${(active === item.key || openMenu === item.key) ? " active" : ""} ${snapshot.isDragging ? 'dragging' : ''}`}
                          >
                            <div className="sidebar-link-left" onDoubleClick={(e) => { e.stopPropagation(); startInlineRename(item.key, item.label); }}>
                              <input
                                type="checkbox"
                                className="sidebar-checkbox"
                                aria-label={`Select ${item.label}`}
                                checked={isSelected}
                                onClick={(e) => {
                                  const checked = (e as unknown as MouseEvent & { target: HTMLInputElement }).target['checked'];
                                  const shift = (e as unknown as MouseEvent).shiftKey;
                                  setSelectedKeys(prev => {
                                    if (shift && lastClickedRef.current) {
                                      const listForRange = displayedPages; // with empty search, this is allNonDashboard
                                      const start = listForRange.findIndex(p => p.key === lastClickedRef.current);
                                      const end = listForRange.findIndex(p => p.key === item.key);
                                      if (start !== -1 && end !== -1) {
                                        const [a, b] = start < end ? [start, end] : [end, start];
                                        const rangeKeys = listForRange.slice(a, b + 1).map(x => x.key);
                                        if (checked) {
                                          return Array.from(new Set([...prev, ...rangeKeys]));
                                        } else {
                                          return prev.filter(k => !rangeKeys.includes(k));
                                        }
                                      }
                                    }
                                    if (checked) return Array.from(new Set([...prev, item.key]));
                                    return prev.filter(k => k !== item.key);
                                  });
                                  lastClickedRef.current = item.key;
                                  // stop propagation so row click doesn't navigate
                                  (e as any).stopPropagation();
                                }}
                              />
                              {editingKey === item.key ? (
                                <input
                                  ref={editingInputRef}
                                  className="sidebar-rename-input"
                                  value={editingValue}
                                  onChange={(e) => setEditingValue(e.target.value)}
                                  onBlur={commitInlineRename}
                                  aria-label={`Rename ${item.label}`}
                                  placeholder="Rename analysis"
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter') { e.preventDefault(); commitInlineRename(); }
                                    else if (e.key === 'Escape') { e.preventDefault(); cancelInlineRename(); }
                                  }}
                                />
                              ) : (
                                <button
                                  className="sidebar-link sidebar-link-button"
                                  aria-label={`Open ${item.label}`}
                                  onClick={() => onNavigate(item.key)}
                                >
                                  {item.key === 'dashboard' ? (
                                    <span className="home-replace-inline"><span className="home-add-icon" aria-hidden>+</span><span className="new-bubble">NEW ANALYSIS</span></span>
                                  ) : (
                                    <>
                                      {item.icon ? (
                                        <span
                                          className={`sidebar-item-icon ${onChangeIcon ? 'clickable' : ''}`}
                                          aria-hidden
                                          onClick={(e) => { e.stopPropagation(); if (onChangeIcon) { setIconPickerKey(item.key); setIconPickerOpen(true); } }}
                                          title="Change icon"
                                        >{item.icon}</span>
                                      ) : (
                                        <span
                                          className={`sidebar-item-icon folder-emoji ${onChangeIcon ? 'clickable' : ''}`}
                                          aria-hidden
                                          onClick={(e) => { e.stopPropagation(); if (onChangeIcon) { setIconPickerKey(item.key); setIconPickerOpen(true); } }}
                                          title="Change icon"
                                        >📁</span>
                                      )}
                                      <span className="sidebar-label">{item.label}</span>
                                    </>
                                  )}
                                </button>
                              )}
                            </div>
                            {/* action: three dots - visible on hover or when active; don't show for dashboard */}
                            {item.key !== 'dashboard' && (
                              <div className="sidebar-actions-wrap">
                                {/* dedicated drag grip placed next to the action button - easier to hit */}
                                <span className="action-grip" {...providedDraggable.dragHandleProps} title="Drag to reorder" aria-hidden>≡</span>
                                <button
                                  className="item-actions"
                                  title="More"
                                  onClick={(e) => { e.stopPropagation(); setOpenMenu(openMenu === item.key ? null : item.key); }}
                                >
                                  &#8230;
                                </button>

                                {openMenu === item.key && (
                                  <div className="item-menu" onClick={(e) => e.stopPropagation()}>
                                    <button className="item-menu-btn" onClick={() => { startInlineRename(item.key, item.label); }}>Rename</button>
                                    <button
                                      className="item-menu-btn danger"
                                      onClick={() => {
                                        askConfirm(`Delete "${item.label}"? This action cannot be undone.`, () => {
                                          onDeletePage && onDeletePage(item.key);
                                          setOpenMenu(null);
                                          setConfirmOpen(false);
                                        });
                                      }}
                                    >Delete</button>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      }}
                    </Draggable>
                  );
                })}
                {provided.placeholder}
                </div>
              )}
            </Droppable>
          </DragDropContext>
        ) : (
          <div>
            {/* bulk action bar shown when there are selected items */}
            {selectedKeys.length > 0 && (
              <div className="sidebar-bulk-bar">
                <div className="bulk-info">{selectedKeys.length} selected</div>
                <div className="bulk-actions">
                  <button className="item-menu-btn" onClick={() => { setSelectedKeys([]); }}>Clear</button>
                  <button
                    className="item-menu-btn danger"
                    onClick={() => {
                      askConfirm(`Delete ${selectedKeys.length} page${selectedKeys.length === 1 ? '' : 's'}? This action cannot be undone.`, () => {
                        if (onBulkDelete) {
                          onBulkDelete(selectedKeys);
                        } else if (onDeletePage) {
                          selectedKeys.forEach(k => onDeletePage(k));
                        }
                        setSelectedKeys([]);
                        setConfirmOpen(false);
                      });
                    }}
                  >Delete</button>
                </div>
              </div>
            )}
            {displayedPages.length === 0 && (
              <div className="no-results">No results</div>
            )}
            {displayedPages.map((item) => {
              const isSelected = selectedKeys.includes(item.key);
              return (
                <div key={item.key} className={`sidebar-item${(active === item.key || openMenu === item.key) ? " active" : ""}`}>
                  <div className="sidebar-link-left" onDoubleClick={(e) => { e.stopPropagation(); startInlineRename(item.key, item.label); }}>
                    <input
                      type="checkbox"
                      className="sidebar-checkbox"
                      aria-label={`Select ${item.label}`}
                      checked={isSelected}
                      onClick={(e) => {
                        const checked = (e as unknown as MouseEvent & { target: HTMLInputElement }).target['checked'];
                        const shift = (e as unknown as MouseEvent).shiftKey;
                        setSelectedKeys(prev => {
                          if (shift && lastClickedRef.current) {
                            const listForRange = displayedPages;
                            const start = listForRange.findIndex(p => p.key === lastClickedRef.current);
                            const end = listForRange.findIndex(p => p.key === item.key);
                            if (start !== -1 && end !== -1) {
                              const [a, b] = start < end ? [start, end] : [end, start];
                              const rangeKeys = listForRange.slice(a, b + 1).map(x => x.key);
                              if (checked) {
                                return Array.from(new Set([...prev, ...rangeKeys]));
                              } else {
                                return prev.filter(k => !rangeKeys.includes(k));
                              }
                            }
                          }
                          if (checked) return Array.from(new Set([...prev, item.key]));
                          return prev.filter(k => k !== item.key);
                        });
                        lastClickedRef.current = item.key;
                        (e as any).stopPropagation();
                      }}
                    />
                    {editingKey === item.key ? (
                      <input
                        ref={editingInputRef}
                        className="sidebar-rename-input"
                        value={editingValue}
                        onChange={(e) => setEditingValue(e.target.value)}
                        onBlur={commitInlineRename}
                        aria-label={`Rename ${item.label}`}
                        placeholder="Rename analysis"
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') { e.preventDefault(); commitInlineRename(); }
                          else if (e.key === 'Escape') { e.preventDefault(); cancelInlineRename(); }
                        }}
                      />
                    ) : (
                      <button
                        className="sidebar-link sidebar-link-button"
                        aria-label={`Open ${item.label}`}
                        onClick={() => onNavigate(item.key)}
                      >
                        {item.icon ? (
                          <span className="sidebar-item-icon" aria-hidden>{item.icon}</span>
                        ) : (
                          <span className="sidebar-item-icon folder-emoji" aria-hidden>📁</span>
                        )}
                        <span className="sidebar-label">{item.label}</span>
                      </button>
                    )}
                  </div>
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
                          <button className="item-menu-btn" onClick={() => { startInlineRename(item.key, item.label); }}>Rename</button>
                          <button
                            className="item-menu-btn danger"
                            onClick={() => {
                              askConfirm(`Delete "${item.label}"? This action cannot be undone.`, () => {
                                onDeletePage && onDeletePage(item.key);
                                setOpenMenu(null);
                                setConfirmOpen(false);
                              });
                            }}
                          >Delete</button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </nav>
  {!collapsed && (
        <div className="sidebar-search-wrap">
          <input
            className="sidebar-search"
            placeholder="Search for..."
            aria-label="Search pages"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Escape') setSearchQuery(''); }}
          />
          {q !== '' && (
            <button className="search-clear" aria-label="Clear search" onClick={() => setSearchQuery('')}>×</button>
          )}
        </div>
      )}

  {/* footer */}
  {collapsed ? (
    <div className="sidebar-footer collapsed-footer">
      <div className="profile-container">
        <button className="profile-btn" title="Profile" aria-label="Profile" onClick={() => setProfileOpen(p => !p)} aria-haspopup="dialog">
          <img src={userProfileIcon} alt="" aria-hidden width="16" height="16" />
          <span className="visually-hidden">Open profile</span>
        </button>
        <ProfilePopup open={profileOpen} onClose={() => setProfileOpen(false)} user={{ name: 'Guest User', email: 'guest@example.com', plan: 'Guest' }} />
      </div>
      <button className="settings-btn" title="Settings" aria-label="Settings" onClick={() => setSettingsOpen(true)}>
        <img src={settingsIcon} alt="" aria-hidden width="16" height="16" />
      </button>
    </div>
  ) : (
    <div className="sidebar-footer">
      <div className="profile-container">
        <button className="profile-btn" title="Profile" aria-label="Profile" onClick={() => setProfileOpen(p => !p)} aria-haspopup="dialog">
          <img src={userProfileIcon} alt="" aria-hidden width="16" height="16" />
          <span className="visually-hidden">Open profile</span>
        </button>
        <ProfilePopup open={profileOpen} onClose={() => setProfileOpen(false)} user={{ name: 'Guest User', email: 'guest@example.com', plan: 'Guest' }} />
      </div>
      <button className="settings-btn" title="Settings" aria-label="Settings" onClick={() => setSettingsOpen(true)}>
        <img src={settingsIcon} alt="" aria-hidden width="16" height="16" />
      </button>
    </div>
  )}
      <ConfirmDialog
        open={confirmOpen}
        title="Confirm deletion"
        message={confirmMessage}
        confirmText="Delete"
        cancelText="Cancel"
        danger
        onConfirm={() => { confirmActionRef.current && confirmActionRef.current(); }}
        onCancel={() => setConfirmOpen(false)}
      />
      {iconPickerOpen && iconPickerKey && (
        <IconPickerModal
          isOpen={iconPickerOpen}
          currentIcon={pages.find(p => p.key === iconPickerKey)?.icon}
          onClose={() => { setIconPickerOpen(false); setIconPickerKey(null); }}
          onSelect={(ic) => {
            if (onChangeIcon && iconPickerKey) {
              onChangeIcon(iconPickerKey, ic);
            }
            setIconPickerOpen(false);
            setIconPickerKey(null);
          }}
        />
      )}
      {settingsOpen && (
        <SettingsModal isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
      )}
    </aside>
  );
};

export default Sidebar;
