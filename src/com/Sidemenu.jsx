import React from "react";
import { Link } from "react-router-dom";
import "../style/sidemenu.css";

const Sidemenu = () => {
  return (
    <>
      <div className="main_menu">
        <div className="list_menu">

            <Link to="/home">
              <li className="li_menu">Home</li>
            </Link>
            <hr />
            <Link to="/about">
              <li className="li_menu">About Me</li>
            </Link>
            <hr />
            <Link to="/project">
              <li className="li_menu">Project</li>
            </Link>
            <hr />
            <Link to="/skill">
              <li className="li_menu">Skill</li>
            </Link>
            <hr />
            <Link to="/contact">
              <li className="li_menu">Contact</li>
            </Link>
            <hr />
        </div>
        <div className="clic_menu">
          <Link to="/contact">
            <button className="clic_me">Hire Me</button>
          </Link>
        </div>
      </div>
    </>
  );
};

export default Sidemenu;
