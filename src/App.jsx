import React from "react";
import { Route, Routes } from "react-router-dom";
import Head from "./com/Head";
import Home from "./com/Home";
import About from "./com/About";
import Skill from "./com/Skill";
import Project from "./com/Project";
import Contact from "./com/Contact";
import "../src/App.css";
import Sidemenu from "./com/Sidemenu";
import Conta from "./com/Conta";

const App = () => {
  return (
    <>
      <div className="head">
        <Head />
      </div>
      <div className="side_menu">
        <Sidemenu />
      </div>
      <Routes>
        <Route className="home" path="/home" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/skill" element={<Skill />} />
        <Route path="/project" element={<Project />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="*" element={<Home />} />
      </Routes>
      <Conta />
    </>
  );
};

export default App;
