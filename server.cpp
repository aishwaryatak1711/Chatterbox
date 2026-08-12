

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// ════════════════════════════════════════════════════════════
//  Hardcoded API credentials
// ════════════════════════════════════════════════════════════
static const std::string GROQ_API_KEY = "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";
static const std::string GROQ_MODEL   = "llama-3.3-70b-versatile";

// ════════════════════════════════════════════════════════════
//  Utility helpers
// ════════════════════════════════════════════════════════════
static std::string jsonEscape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else o += (char)c;
    }
    return o;
}

// Extract first occurrence of "key":"value" from JSON string
static std::string extractJson(const std::string& json, const std::string& key) {
    std::string k = "\"" + key + "\"";
    size_t p = json.find(k);
    if (p == std::string::npos) return "";
    size_t q = json.find(':', p + k.size());
    if (q == std::string::npos) return "";
    size_t s = json.find('"', q + 1);
    if (s == std::string::npos) return "";
    std::string val; bool esc = false;
    for (size_t i = s+1; i < json.size(); ++i) {
        char c = json[i];
        if (esc) { esc=false; val += (c=='n'?'\n': c=='t'?'\t': c); }
        else if (c=='\\') esc=true;
        else if (c=='"')  break;
        else val += c;
    }
    return val;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b-a+1);
}

// ════════════════════════════════════════════════════════════
//  CLASS 1: Message  — Encapsulation
//  Stores a single chat turn (role + content + timestamp).
//  All fields are private; access only through getters.
// ════════════════════════════════════════════════════════════
class Message {
private:
    std::string role_;       // "user" or "model"
    std::string content_;
    std::time_t timestamp_;

public:
    Message(const std::string& role, const std::string& content)
        : role_(role), content_(content), timestamp_(std::time(nullptr)) {}

    // Getters — controlled access (Encapsulation)
    const std::string& role()    const { return role_;    }
    const std::string& content() const { return content_; }

    std::string formattedTime() const {
        char buf[16];
        std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&timestamp_));
        return buf;
    }

    // Groq/OpenAI API format: {"role":"user","content":"..."}
    std::string toGroqJSON() const {
        return "{\"role\":\"" + role_ + "\","
               "\"content\":\"" + jsonEscape(content_) + "\"}";
    }
};

// ════════════════════════════════════════════════════════════
//  CLASS 2: ConversationHistory  — Encapsulation
//  Manages the sliding window of messages sent to the API.
// ════════════════════════════════════════════════════════════
class ConversationHistory {
private:
    std::vector<Message> msgs_;
    size_t maxMessages_;

    void prune() {
        while (msgs_.size() > maxMessages_)
            msgs_.erase(msgs_.begin());
    }

public:
    explicit ConversationHistory(size_t max = 30)
        : maxMessages_(max) {}

    void add(const Message& m) { msgs_.push_back(m); prune(); }

    void clear() { msgs_.clear(); }

    // Build Gemini "contents" JSON array
    std::string toGroqMessages() const {
        std::string out = "[";
        bool first = true;
        for (const auto& m : msgs_) {
            if (m.role() == "user" || m.role() == "model") {
                if (!first) out += ",";
                out += m.toGroqJSON();
                first = false;
            }
        }
        out += "]";
        return out;
    }

    size_t size()  const { return msgs_.size(); }
    bool   empty() const { return msgs_.empty(); }
    const std::vector<Message>& messages() const { return msgs_; }
};

// ════════════════════════════════════════════════════════════
//  CLASS 3: AIProvider  — Abstraction (pure abstract class)
//  Defines the interface any AI backend must satisfy.
//  Callers never see HTTP details — only sendRequest().
// ════════════════════════════════════════════════════════════
class AIProvider {
public:
    virtual ~AIProvider() = default;
    virtual std::string sendRequest(const ConversationHistory& history,
                                    const std::string& systemPrompt) = 0;
    virtual std::string name() const = 0;
};

// ════════════════════════════════════════════════════════════
//  CLASS 4: GroqProvider  — Inheritance + Polymorphism
//  Inherits AIProvider, overrides sendRequest() to call
//  the Groq REST API (OpenAI-compatible) using curl.
// ════════════════════════════════════════════════════════════
class GroqProvider : public AIProvider {   // ← Inheritance
private:
    std::string apiKey_;
    std::string model_;

    // Build Groq/OpenAI-compatible request body
    std::string buildBody(const ConversationHistory& history,
                          const std::string& sysPrompt) const {
        std::string msgs = "[";
        bool first = true;
        if (!sysPrompt.empty()) {
            msgs += "{\"role\":\"system\",\"content\":\"" + jsonEscape(sysPrompt) + "\"}";
            first = false;
        }
        for (const auto& m : history.messages()) {
            if (!first) msgs += ",";
            msgs += m.toGroqJSON();
            first = false;
        }
        msgs += "]";
        return "{\"model\":\"" + model_ + "\","
               "\"messages\":" + msgs + ","
               "\"temperature\":0.7,"
               "\"max_tokens\":1024}";
    }

    std::string callAPI(const std::string& body) const {
        std::string tmpIn  = "/tmp/nexus_req.json";
        std::string tmpOut = "/tmp/nexus_res.json";

        FILE* f = fopen(tmpIn.c_str(), "w");
        if (!f) throw std::runtime_error("Cannot write temp file");
        fputs(body.c_str(), f);
        fclose(f);

        std::string cmd =
            "curl -s -X POST https://api.groq.com/openai/v1/chat/completions "
            "-H \"Content-Type: application/json\" "
            "-H \"Authorization: Bearer " + apiKey_ + "\" "
            "-d @" + tmpIn + " -o " + tmpOut + " 2>/dev/null";

        if (std::system(cmd.c_str()) != 0)
            throw std::runtime_error("curl failed — is curl installed?");

        FILE* r = fopen(tmpOut.c_str(), "r");
        if (!r) throw std::runtime_error("Cannot read API response");
        std::string res; char buf[512];
        while (fgets(buf, sizeof(buf), r)) res += buf;
        fclose(r);
        return res;
    }

public:
    GroqProvider(const std::string& apiKey,
                   const std::string& model = "llama-3.3-70b-versatile")
        : apiKey_(apiKey), model_(model) {}

    std::string name() const override {          // ← Polymorphism
        return "Groq / " + model_;
    }

    std::string sendRequest(const ConversationHistory& history,
                            const std::string& sysPrompt) override {   // ← Polymorphism
        std::string body     = buildBody(history, sysPrompt);
        std::string response = callAPI(body);

        // Check for API error
        std::string errMsg = extractJson(response, "message");
        if (!errMsg.empty() && response.find("\"error\"") != std::string::npos)
            throw std::runtime_error("Groq API error: " + errMsg);

        // OpenAI-compatible: choices[0].message.content
        std::string text = extractJson(response, "content");
        if (text.empty())
            throw std::runtime_error("Empty or unexpected Groq response");
        return text;
    }
};

// ════════════════════════════════════════════════════════════
//  CLASS 5: DemoProvider  — Inheritance + Polymorphism
//  Offline fallback — no network needed.
// ════════════════════════════════════════════════════════════
class DemoProvider : public AIProvider {     // ← Inheritance
private:
    int turn_ = 0;
    std::string reply(const std::string& in) {
        ++turn_;
        std::string l = in;
        std::transform(l.begin(), l.end(), l.begin(), ::tolower);
        if (l.find("hello") != std::string::npos) return "Hello! (Demo mode)";
        if (l.find("oop")   != std::string::npos) return "OOP: Encapsulation, Inheritance, Polymorphism, Abstraction.";
        if (l.find("c++")   != std::string::npos) return "C++ supports full OOP with classes and virtual functions.";
        return "Interesting! (Demo mode — Gemini key is hardcoded for real responses.)";
    }
public:
    std::string name() const override { return "Demo"; }   // ← Polymorphism
    std::string sendRequest(const ConversationHistory& h,
                            const std::string&) override { // ← Polymorphism
        if (h.empty()) return "Hello!";
        for (int i=(int)h.messages().size()-1; i>=0; --i)
            if (h.messages()[i].role() == "user")
                return reply(h.messages()[i].content());
        return "Please say something!";
    }
};

// ════════════════════════════════════════════════════════════
//  CLASS 6: Chatbot  — Façade (Encapsulation + Polymorphism)
//  Hides history management and provider calls behind chat().
// ════════════════════════════════════════════════════════════
class Chatbot {
private:
    std::string                 name_;
    std::string                 systemPrompt_;
    ConversationHistory         history_;
    std::unique_ptr<AIProvider> provider_;
    int                         turns_ = 0;

public:
    Chatbot(const std::string& name,
            const std::string& sysPrompt,
            std::unique_ptr<AIProvider> provider)
        : name_(name), systemPrompt_(sysPrompt),
          provider_(std::move(provider)) {}

    std::string chat(const std::string& userInput) {
        history_.add(Message("user", userInput));
        // Polymorphic dispatch — calls GroqProvider or DemoProvider
        std::string reply = provider_->sendRequest(history_, systemPrompt_);
        history_.add(Message("model", reply));
        ++turns_;
        return reply;
    }

    void reset() { history_.clear(); turns_ = 0; }

    const std::string& botName()      const { return name_;  }
    int                turns()        const { return turns_; }
    std::string        providerName() const { return provider_->name(); }
    size_t             historySize()  const { return history_.size(); }
};

// ════════════════════════════════════════════════════════════
//  CLASS 7: HttpServer  — Encapsulation
//  All POSIX socket code is hidden inside this class.
// ════════════════════════════════════════════════════════════
class HttpServer {
private:
    int     port_;
    int     fd_ = -1;
    Chatbot& bot_;

    static std::string httpResp(int code, const std::string& ct,
                                 const std::string& body) {
        std::string st = code==200 ? "OK" : code==400 ? "Bad Request" : "Server Error";
        std::ostringstream r;
        r << "HTTP/1.1 " << code << " " << st << "\r\n"
          << "Content-Type: " << ct << "\r\n"
          << "Access-Control-Allow-Origin: *\r\n"
          << "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
          << "Access-Control-Allow-Headers: Content-Type\r\n"
          << "Content-Length: " << body.size() << "\r\n"
          << "Connection: close\r\n\r\n" << body;
        return r.str();
    }

    static std::string getBody(const std::string& req) {
        size_t p = req.find("\r\n\r\n");
        return p == std::string::npos ? "" : req.substr(p+4);
    }

    void handle(int cfd) {
        char buf[8192] = {};
        recv(cfd, buf, sizeof(buf)-1, 0);
        std::string req(buf);

        if (req.substr(0,7) == "OPTIONS") {
            auto r = httpResp(200,"text/plain","");
            send(cfd,r.c_str(),r.size(),0); close(cfd); return;
        }
        if (req.find("GET /health") != std::string::npos) {
            std::string b = "{\"status\":\"ok\",\"provider\":\"" +
                            jsonEscape(bot_.providerName()) + "\"}";
            auto r = httpResp(200,"application/json",b);
            send(cfd,r.c_str(),r.size(),0); close(cfd); return;
        }
        if (req.find("POST /reset") != std::string::npos) {
            bot_.reset();
            auto r = httpResp(200,"application/json","{\"status\":\"reset\"}");
            send(cfd,r.c_str(),r.size(),0); close(cfd); return;
        }
        if (req.find("POST /chat") != std::string::npos) {
            std::string body = getBody(req);
            std::string msg  = trim(extractJson(body, "message"));
            if (msg.empty()) {
                auto r = httpResp(400,"application/json","{\"error\":\"Missing message\"}");
                send(cfd,r.c_str(),r.size(),0); close(cfd); return;
            }
            try {
                std::string reply = bot_.chat(msg);
                std::string jb = "{\"reply\":\"" + jsonEscape(reply) +
                                 "\",\"turns\":" + std::to_string(bot_.turns()) + "}";
                auto r = httpResp(200,"application/json",jb);
                send(cfd,r.c_str(),r.size(),0);
            } catch (const std::exception& ex) {
                std::string jb = "{\"error\":\"" + jsonEscape(ex.what()) + "\"}";
                auto r = httpResp(500,"application/json",jb);
                send(cfd,r.c_str(),r.size(),0);
            }
            close(cfd); return;
        }
        auto r = httpResp(404,"application/json","{\"error\":\"Not found\"}");
        send(cfd,r.c_str(),r.size(),0); close(cfd);
    }

public:
    HttpServer(int port, Chatbot& bot) : port_(port), bot_(bot) {}
    ~HttpServer() { if (fd_ >= 0) close(fd_); }

    void start() {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) throw std::runtime_error("socket() failed");
        int opt=1; setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(port_);

        if (bind(fd_, (sockaddr*)&addr, sizeof(addr)) < 0)
            throw std::runtime_error("bind() failed — port busy?");
        listen(fd_, 16);

        std::cout << "\n  ┌──────────────────────────────────────────┐\n"
                  << "  │  NexusBot C++ Server — port " << port_ << "          │\n"
                  << "  │  Provider : " << bot_.providerName()
                  << std::string(30 - (int)bot_.providerName().size(),' ') << "│\n"
                  << "  │  API Key  : hardcoded                    │\n"
                  << "  │  POST /chat  GET /health  POST /reset    │\n"
                  << "  └──────────────────────────────────────────┘\n\n";

        while (true) {
            sockaddr_in ca{}; socklen_t cl = sizeof(ca);
            int cfd = accept(fd_, (sockaddr*)&ca, &cl);
            if (cfd < 0) continue;
            std::cout << "  → " << inet_ntoa(ca.sin_addr) << "\n";
            handle(cfd);
        }
    }
};

// ════════════════════════════════════════════════════════════
//  main — wires everything together
// ════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    int port = (argc > 1) ? std::stoi(argv[1]) : 8080;

    // API key is hardcoded — no env var needed
    auto provider = std::make_unique<GroqProvider>(GROQ_API_KEY, GROQ_MODEL);

    Chatbot bot(
        "NexusBot",
        "You are NexusBot, a friendly and knowledgeable AI assistant. "
        "Give clear, concise, accurate answers. Use code blocks for code.",
        std::move(provider)
    );

    try {
        HttpServer server(port, bot);
        server.start();
    } catch (const std::exception& ex) {
        std::cerr << "Fatal: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
