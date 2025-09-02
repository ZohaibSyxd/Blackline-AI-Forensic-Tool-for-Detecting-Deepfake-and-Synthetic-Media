
import React, { useState } from "react";
import { FaHome, FaFolderOpen, FaCog, FaBell, FaUserCircle } from "react-icons/fa";
import "./Sidebar.css";

interface SidebarProps {
  active: string;
  onNavigate: (page: string) => void;
  onAddPage?: () => void;
  onDeletePage?: (key: string) => void;
  onRenamePage?: (key: string) => void;
  pages: { key: string; label: string }[];
}

// pages are provided by the parent (App) so the sidebar reflects dynamic additions

const Sidebar: React.FC<SidebarProps> = ({ active, onNavigate, onAddPage, onDeletePage, onRenamePage, pages }) => {
  const [openMenu, setOpenMenu] = useState<string | null>(null);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="logo">WebbyFrames <span className="for-figma">for Figma</span></span>
      </div>
      <div className="sidebar-profile">
        <div className="profile-left">
          <FaUserCircle size={36} />
        </div>
        <div className="profile-actions">
          <FaCog className="sidebar-icon" />
          <div className="sidebar-notification">
            <FaBell />
          </div>
          <button className="add-page-btn" title="Add page" onClick={() => onAddPage && onAddPage()}>+</button>
        </div>
      </div>
  <nav className="sidebar-nav">
        {pages.map((item) => (
          <div
            key={item.key}
            className={`sidebar-item${active === item.key ? " active" : ""}`}
            onMouseLeave={() => setOpenMenu(null)}
          >
            <div className="sidebar-link" onClick={() => onNavigate(item.key)} role="button" tabIndex={0}>
              <div className="sidebar-link-left">{item.key === 'dashboard' ? <FaHome /> : <FaFolderOpen />} {item.label}</div>
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
