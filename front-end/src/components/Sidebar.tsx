
import { FaHome, FaFolderOpen, FaCog, FaBell, FaUserCircle } from "react-icons/fa";
import "./Sidebar.css";

interface SidebarProps {
  active: string;
  onNavigate: (page: string) => void;
}

const navItems = [
  { key: "dashboard", icon: <FaHome />, label: "HOME PAGE" },
  { key: "file2", icon: <FaFolderOpen />, label: "FILE ANALYSIS 2" },
  { key: "file1", icon: <FaFolderOpen />, label: "FILE ANALYSIS 1" },
  { key: "file3", icon: <FaFolderOpen />, label: "FILE ANALYSIS 3" },
  { key: "file$", icon: <FaFolderOpen />, label: "FILE ANALYSIS $" },
];

const Sidebar: React.FC<SidebarProps> = ({ active, onNavigate }) => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="logo">WebbyFrames <span className="for-figma">for Figma</span></span>
      </div>
      <div className="sidebar-profile">
        <FaUserCircle size={36} />
        <FaCog className="sidebar-icon" />
        <div className="sidebar-notification">
          <FaBell />
          <span className="badge">9</span>
        </div>
      </div>
      <input className="sidebar-search" placeholder="Search for..." />
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <button
            key={item.key}
            className={`sidebar-link sidebar-link-button${active === item.key ? " active" : ""}`}
            onClick={() => onNavigate(item.key)}
          >
            {item.icon} {item.label}
          </button>
        ))}
      </nav>
    </aside>
  );
};

export default Sidebar;
