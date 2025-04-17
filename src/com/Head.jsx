import React, { useState } from "react";
import "../style/navbar.css";
import { Link } from "react-router-dom";
import menu from "../image/bar.png";

const Head = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <div className="main">
      <div className="portfolio">
        <h1>PORTFOLIO</h1>
      </div>

      {/* Desktop Navigation */}
      <div className="list">
        <ul>
          <Link to="/home">
            <li className="li">Home</li>
          </Link>
          <Link to="/about">
            <li className="li">About Me</li>
          </Link>
          <Link to="/project">
            <li className="li">Project</li>
          </Link>
          <Link to="/skill">
            <li className="li">Skill</li>
          </Link>
          <Link to="/contact">
            <li className="li">Contact</li>
          </Link>
          <Link to="/contact">
            <button className="clic">Hire Me</button>
          </Link>
        </ul>
      </div>

      {/* Mobile Menu Button */}
      <div className="menu_bar">
        <img
          src={menu}
          alt="Menu"
          onClick={toggleMenu}
          className={`menu-icon ${isMenuOpen ? "active" : ""}`}
        />
      </div>

      {/* Mobile Menu Overlay */}
      <div className={`mobile-menu ${isMenuOpen ? "active" : ""}`}>
        <ul>
          <Link to="/home" onClick={toggleMenu}>
            <li>Home</li>
          </Link>
          <Link to="/about" onClick={toggleMenu}>
            <li>About Me</li>
          </Link>
          <Link to="/project" onClick={toggleMenu}>
            <li>Project</li>
          </Link>
          <Link to="/skill" onClick={toggleMenu}>
            <li>Skill</li>
          </Link>
          <Link to="/contact" onClick={toggleMenu}>
            <li>Contact</li>
          </Link>
          <Link to="/contact" onClick={toggleMenu}>
            <button className="mobile-clic">Hire Me</button>
          </Link>
        </ul>
      </div>
    </div>
  );
};

export default Head;
