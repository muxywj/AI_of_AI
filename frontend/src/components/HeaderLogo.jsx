// src/components/HeaderLogo.jsx
import React from "react";
import { useNavigate } from "react-router-dom";

const HeaderLogo = () => {
  const navigate = useNavigate();

  return (
    <h1
      className="text-xl font-semibold cursor-pointer hover:opacity-80 transition-opacity"
      style={{ color: "#5d7c5b", fontWeight: 800 }}
      onClick={() => navigate("/")}
    >
      AI OF AI
    </h1>
  );
};

export default HeaderLogo;