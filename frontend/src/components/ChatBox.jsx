import React, { useState, useEffect, useRef } from "react";
import { Send, CirclePlus, Image as ImageIcon, File as FileIcon, X } from "lucide-react";
import { useChat } from "../context/ChatContext";

// 이미지가 아닌 일반 파일 허용 확장자 목록 (필요시 추가/수정)
const ALLOWED_FILE_EXTS = [
  ".pdf", ".txt", ".csv", ".md",
  ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
  ".zip", ".rar", ".json"
];

const ChatBox = () => {
  const {
    messages = {},
    sendMessage,
    isLoading,
    selectedModels = [],
    // 선택 모델/워크플로우 기능 없음
    processImageUpload,     // 있으면 우선 사용
    processFileUpload       // 있으면 우선 사용
  } = useChat() || {};

  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRefs = useRef({});

  // 첨부(이미지/파일) 상태
  const [imageAttachments, setImageAttachments] = useState([]); // { id, file, url }
  const [fileAttachments, setFileAttachments] = useState([]);   // { id, file, name, size }
  const imageInputRef = useRef(null);
  const fileInputRef = useRef(null);

  // + 버튼 메뉴
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const plusBtnRef = useRef(null);

  // 메시지 컬럼별 끝 ref 준비
  useEffect(() => {
    selectedModels.concat("optimal").forEach((modelId) => {
      if (!messagesEndRefs.current[modelId]) {
        messagesEndRefs.current[modelId] = React.createRef();
      }
    });
  }, [selectedModels]);

  // 새 메시지 추가 시 자동 스크롤
  useEffect(() => {
    selectedModels.concat("optimal").forEach((modelId) => {
      messagesEndRefs.current[modelId]?.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages, selectedModels]);

  // 바깥 클릭으로 메뉴 닫기
  useEffect(() => {
    const onDocClick = (e) => {
      if (!isMenuOpen) return;
      const menuEl = menuRef.current;
      const btnEl = plusBtnRef.current;
      if (menuEl && btnEl && !menuEl.contains(e.target) && !btnEl.contains(e.target)) {
        setIsMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [isMenuOpen]);

  const generateId = () => `att-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
  const generateRequestId = () => `req-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;

  // File -> Base64 dataURL
  const readFileAsDataURL = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = reject;
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(file);
    });

  // 이미지 onChange (이미지만 통과)
  const handleImageChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    if (!file.type?.startsWith("image/")) {
      alert("이미지 파일만 선택할 수 있어요.");
      e.target.value = "";
      return;
    }

    const url = URL.createObjectURL(file);
    setImageAttachments((prev) => [...prev, { id: generateId(), file, url }]);

    // 같은 파일 다시 선택해도 change 발생하도록 초기화
    try { e.target.value = ""; } catch {}
    setIsMenuOpen(false);
  };

  // 파일 onChange (이미지 제외)
  const handleFileChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    if (file.type?.startsWith("image/")) {
      alert("파일 업로드에서는 이미지가 아닌 파일만 선택할 수 있어요.");
      e.target.value = "";
      return;
    }

    // 확장자 제한(선택 사항) — accept로 1차 필터링하지만 JS에서도 2차 방어
    const lowerName = file.name.toLowerCase();
    const allowed = ALLOWED_FILE_EXTS.some(ext => lowerName.endsWith(ext));
    if (!allowed) {
      alert(`허용되지 않는 파일 형식입니다. 허용: ${ALLOWED_FILE_EXTS.join(", ")}`);
      e.target.value = "";
      return;
    }

    setFileAttachments((prev) => [
      ...prev,
      { id: generateId(), file, name: file.name, size: file.size },
    ]);
    try { e.target.value = ""; } catch {}
    setIsMenuOpen(false);
  };

  // 첨부 제거
  const removeImage = (id) => {
    setImageAttachments((prev) => {
      const target = prev.find((p) => p.id === id);
      if (target?.url) {
        try { URL.revokeObjectURL(target.url); } catch {}
      }
      return prev.filter((p) => p.id !== id);
    });
  };
  const removeFile = (id) => {
    setFileAttachments((prev) => prev.filter((p) => p.id !== id));
  };

  // 전송
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!sendMessage) return;

    const trimmed = inputMessage.trim();
    const hasAttachments = imageAttachments.length > 0 || fileAttachments.length > 0;
    if (!trimmed && !hasAttachments) return;

    const requestId = generateRequestId();

    try {
      // 1) 업로드 핸들러가 있으면 그쪽 경로를 우선 사용 (서버에 파일 전송)
      if (typeof processImageUpload === "function" || typeof processFileUpload === "function") {
        // 이미지는 개별 업로드
        if (typeof processImageUpload === "function") {
          for (const att of imageAttachments) {
            await processImageUpload(att.file, requestId, { caption: trimmed || "" });
          }
        }
        // 파일은 개별 업로드
        if (typeof processFileUpload === "function") {
          for (const att of fileAttachments) {
            await processFileUpload(att.file, requestId, { caption: trimmed || "" });
          }
        }
        // 텍스트만 남았으면 전송
        if (trimmed) {
          await sendMessage(trimmed, requestId, {});
        }
      } else {
        // 2) 폴백: Base64로 변환해서 옵션으로 전달
        const imagesBase64 = await Promise.all(
          imageAttachments.map(async (a) => {
            const dataUrl = await readFileAsDataURL(a.file); // "data:image/png;base64,...."
            return { name: a.file.name, type: a.file.type, size: a.file.size, dataUrl };
          })
        );
        const filesBase64 = await Promise.all(
          fileAttachments.map(async (a) => {
            const dataUrl = await readFileAsDataURL(a.file);
            return { name: a.file.name, type: a.file.type, size: a.file.size, dataUrl };
          })
        );

        // 텍스트에 첨부 메타를 추가(서버가 options를 무시해도 인지 가능)
        const attachmentNote = [
          ...imageAttachments.map(a => `📷 ${a.file.name}`),
          ...fileAttachments.map(a => `📎 ${a.file.name}`)
        ];
        const textWithNote =
          trimmed || (attachmentNote.length ? `(첨부 전송) ${attachmentNote.join(", ")}` : "");

        await sendMessage(textWithNote, requestId, {
          imagesBase64,
          filesBase64,
        });
      }

      // 초기화
      imageAttachments.forEach((a) => {
        if (a.url) try { URL.revokeObjectURL(a.url); } catch {}
      });
      setImageAttachments([]);
      setFileAttachments([]);
      setInputMessage("");
    } catch (err) {
      console.error(err);
      // 실패 시에도 첨부 유지 (사용자가 다시 시도 가능)
    }
  };

  const loadingText = isLoading ? "분석중…" : "";

  return (
    <div className="h-full w-full flex flex-col" style={{ background: "rgba(245, 242, 234, 0.4)" }}>
      <style jsx>{`
        .chat-header {
          background: rgba(245, 242, 234, 0.4);
          backdrop-filter: blur(10px);
          border-bottom: 1px solid rgba(139, 168, 138, 0.15);
          height: 60px;
        }
        .chat-column {
          background: rgba(255, 255, 255, 0.3);
          backdrop-filter: blur(5px);
        }
        .chat-container {
          height: calc(100% - 180px);
        }
        .aiofai-input-area {
          background: rgba(245, 242, 234, 0.4);
          backdrop-filter: blur(10px);
          border-top: 1px solid rgba(139, 168, 138, 0.15);
          padding: 0.75rem 1.2rem;
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.3rem; /* 위/아래 균형 */
        }
        .aiofai-user-message {
          background: linear-gradient(135deg, #5d7c5b, #8ba88a);
          color: #ffffff;
          padding: 1.2rem 1.5rem;
          border-radius: 24px 24px 8px 24px;
          max-width: 85%;
          box-shadow: 0 8px 32px rgba(93, 124, 91, 0.3);
          font-weight: 500;
          line-height: 1.5;
          position: relative;
        }
        .aiofai-bot-message {
          background: rgba(255, 255, 255, 0.8);
          backdrop-filter: blur(10px);
          color: #2d3e2c;
          border: 1px solid rgba(139, 168, 138, 0.2);
          padding: 1.2rem 1.5rem;
          border-radius: 24px 24px 24px 8px;
          max-width: 85%;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
          line-height: 1.6;
          position: relative;
        }
        .aiofai-input-box {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          display: flex;
          align-items: center;
          padding: 0.4rem;
          gap: 0.4rem;
          max-width: 51.2rem;
          margin: 0 auto;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          width: 90%;
          position: relative;
        }
        .aiofai-input-box:focus-within {
          border-color: #8ba88a;
          box-shadow: 0 0 0 3px rgba(93, 124, 91, 0.1);
        }
        .input-field {
          flex: 1;
          border: none;
          outline: none;
          padding: 0.6rem;
          background: transparent;
          color: #2d3e2c;
          font-size: 1rem;
          border-radius: 12px;
        }
        .input-field::placeholder {
          color: rgba(45, 62, 44, 0.5);
        }
        .aiofai-icon-button {
          color: #2d3e2c;
          padding: 8px;
          border-radius: 10px;
          transition: all 0.2s ease;
          cursor: pointer;
          border: none;
          background: transparent;
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }
        .aiofai-icon-button:hover {
          background: rgba(139, 168, 138, 0.12);
        }
        .aiofai-icon-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .attachment-strip {
          width: 90%;
          max-width: 51.2rem;
          margin: 0 auto;
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .attachment-chip {
          position: relative;
          display: inline-flex;
          align-items: center;
          gap: 8px;
          border: 1px solid rgba(139, 168, 138, 0.3);
          background: rgba(255, 255, 255, 0.85);
          backdrop-filter: blur(6px);
          border-radius: 12px;
          padding: 6px 10px 6px 6px;
        }
        .attachment-thumb {
          width: 56px;
          height: 56px;
          border-radius: 8px;
          object-fit: cover;
          border: 1px solid rgba(139, 168, 138, 0.25);
        }
        .chip-close {
          position: absolute;
          top: -8px;
          right: -8px;
          width: 22px;
          height: 22px;
          border-radius: 9999px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: white;
          border: 1px solid rgba(139, 168, 138, 0.3);
          box-shadow: 0 2px 8px rgba(0,0,0,0.08);
          cursor: pointer;
        }
        .chip-close:hover {
          background: rgba(255,255,255,0.9);
        }
        .file-label {
          max-width: 220px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          color: #2d3e2c;
          font-size: 0.9rem;
        }
        .plus-menu {
          position: absolute;
          bottom: 52px;
          right: 8px;
          min-width: 180px;
          background: rgba(255,255,255,0.98);
          border: 1px solid rgba(139,168,138,0.25);
          border-radius: 12px;
          box-shadow: 0 8px 28px rgba(0,0,0,0.12);
          padding: 6px;
          z-index: 50;
        }
        .plus-menu button {
          width: 100%;
          text-align: left;
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 10px;
          border-radius: 10px;
          border: none;
          background: transparent;
          color: #2d3e2c;
          cursor: pointer;
        }
        .plus-menu button:hover {
          background: rgba(139,168,138,0.12);
        }
      `}</style>

      {/* 상단 모델 라벨만 유지 */}
      <div className="flex-shrink-0 flex chat-header w-full">
        {selectedModels.concat("optimal").map((modelId) => (
          <div
            key={modelId}
            className="px-4 py-2 text-lg font-semibold text-center border-r flex-1 whitespace-nowrap overflow-hidden text-ellipsis flex items-center justify-center"
            style={{ color: "#2d3e2c", borderRightColor: "rgba(139, 168, 138, 0.3)" }}
          >
            {modelId === "optimal" ? "최적의 답변" : modelId.toUpperCase()}
          </div>
        ))}
      </div>

      {/* 채팅 메시지 영역 */}
      <div
        className="chat-container grid overflow-hidden"
        style={{ gridTemplateColumns: `repeat(${selectedModels.length + 1}, minmax(0, 1fr))` }}
      >
        {selectedModels.concat("optimal").map((modelId) => (
          <div key={modelId} className="border-r flex-1 overflow-y-auto chat-column">
            <div className="h-full px-4 py-3">
              {messages[modelId]?.map((message, index) => {
                const isUser = !!message.isUser;
                return (
                  <div key={`${modelId}-${index}`} className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
                    <div className={`${isUser ? "aiofai-user-message" : "aiofai-bot-message"}`}>
                      <div>{message.text}</div>
                    </div>
                  </div>
                );
              })}

              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">
                    {loadingText || "입력 중..."}
                  </div>
                </div>
              )}

              <div className="h-3" />
              <div ref={messagesEndRefs.current[modelId]} />
            </div>
          </div>
        ))}
      </div>

      {/* 입력/첨부 영역 */}
      <div className="aiofai-input-area">
        {/* 첨부 프리뷰 스트립 */}
        {(imageAttachments.length > 0 || fileAttachments.length > 0) && (
          <div className="attachment-strip">
            {imageAttachments.map((att) => (
              <div key={att.id} className="attachment-chip">
                <img src={att.url} alt="attachment" className="attachment-thumb" />
                <button type="button" className="chip-close" aria-label="이미지 제거" onClick={() => removeImage(att.id)}>
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
            {fileAttachments.map((att) => (
              <div key={att.id} className="attachment-chip">
                <FileIcon className="w-5 h-5" />
                <span className="file-label" title={att.name}>{att.name}</span>
                <button type="button" className="chip-close" aria-label="파일 제거" onClick={() => removeFile(att.id)}>
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* 숨겨진 파일 입력들 — accept로 1차 필터링 */}
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          style={{ display: "none" }}
        />
        <input
          ref={fileInputRef}
          type="file"
          // 이미지 제외한 확장자만 명시
          accept={ALLOWED_FILE_EXTS.join(",")}
          onChange={handleFileChange}
          style={{ display: "none" }}
        />

        {/* 입력 박스 */}
        <form onSubmit={handleSendMessage} className="aiofai-input-box">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="메시지를 입력하세요..."
            className="input-field"
            disabled={isLoading}
          />

          {/* + 버튼 (메뉴 토글) */}
          <button
            type="button"
            ref={plusBtnRef}
            className="aiofai-icon-button"
            onClick={() => setIsMenuOpen((v) => !v)}
            aria-haspopup="menu"
            aria-expanded={isMenuOpen}
            title="첨부 추가"
            disabled={isLoading}
          >
            <CirclePlus className="w-5 h-5" />
          </button>

          {/* 전송 버튼 */}
          <button
            type="submit"
            disabled={
              isLoading ||
              (!inputMessage.trim() && imageAttachments.length === 0 && fileAttachments.length === 0)
            }
            className="aiofai-icon-button"
            title="전송"
          >
            <Send className="w-5 h-5" />
          </button>

          {/* + 메뉴 팝오버 */}
          {isMenuOpen && (
            <div className="plus-menu" ref={menuRef} role="menu">
              <button type="button" onClick={() => imageInputRef.current?.click()} role="menuitem">
                <ImageIcon className="w-4 h-4" />
                이미지 업로드
              </button>
              <button type="button" onClick={() => fileInputRef.current?.click()} role="menuitem">
                <FileIcon className="w-4 h-4" />
                파일 업로드
              </button>
            </div>
          )}
        </form>
      </div>
    </div>
  );
};

export default ChatBox;