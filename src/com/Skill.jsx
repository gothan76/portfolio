import React from "react";
import "../style/skill.css";
import Conta from '../com/Conta.jsx'
import { Link } from "react-router-dom";

const Skill = () => {
  return (
    <div className="main_skill">
      <div className="about_box">
        <h1>Skill</h1>
        <p>
          <Link to="/home">Home</Link> || <a href="#">Skill</a>
        </p>
      </div>

      <div className="skil">
        <li>
          <h3>HTML</h3>
          <span className="bar">
            <span className="html"></span>
          </span>
        </li>
        <li>
          <h3>CSS</h3>
          <span className="bar">
            <span className="css"></span>
          </span>
        </li>
        <li>
          <h3>JAVA SCRIPT</h3>
          <span className="bar">
            <span className="js"></span>
          </span>
        </li>
        <li>
          <h3>REACT</h3>
          <span className="bar">
            <span className="react"></span>
          </span>
        </li>
      </div>
    </div>
  );
};

export default Skill;
