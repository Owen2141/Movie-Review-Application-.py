import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import pandas as pd
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from app import get_imdb_id, get_reviews, analyze_reviews, build_chart, save_text_summary


# -----------------------------
# SEARCH HANDLER
# -----------------------------
def search_movie():
    movie_name = entry.get().strip()

    if not movie_name or movie_name.startswith("e.g."):
        messagebox.showerror("Error", "Please enter a movie name")
        return

    btn.config(state="disabled", text="Searching...")
    status_label.config(text="Looking up movie...")

    def run():
        try:
            tmdb_id, title, rating = get_imdb_id(movie_name)

            if not tmdb_id:
                root.after(0, lambda: messagebox.showerror("Error", "Movie not found on TMDB"))
                root.after(0, reset_btn)
                return

            root.after(0, lambda: status_label.config(
                text=f"Found: {title}  |  TMDB Rating: {rating}  |  Fetching reviews..."
            ))

            reviews = get_reviews(tmdb_id)

            if not reviews:
                root.after(0, lambda: messagebox.showerror(
                    "Error", "No reviews found for this movie on TMDB."
                ))
                root.after(0, reset_btn)
                return

            data = analyze_reviews(reviews)
            df = pd.DataFrame(data, columns=[
                "Review", "Author", "Rating", "Sentiment", "Tokens", "Similarity"
            ])

            root.after(0, lambda: update_ui(df, title, rating))

        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Error", str(e)))
            root.after(0, reset_btn)

    threading.Thread(target=run, daemon=True).start()


# -----------------------------
# UPDATE TABLE + CHART
# -----------------------------
def update_ui(df, title, tmdb_rating):
    global chart_canvas

    for row in tree.get_children():
        tree.delete(row)

    for _, row in df.iterrows():
        tree.insert("", "end", values=(
            row["Author"],
            row["Rating"],
            row["Sentiment"],
            row["Tokens"],
            row["Similarity"],
            row["Review"][:80]
        ))

    status_label.config(
        text=f"{title}  |  TMDB Rating: {tmdb_rating}  |  {len(df)} reviews analysed"
    )

    # CSV → output folder
    os.makedirs("output", exist_ok=True)
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join("output", f"{safe_title}_{timestamp}.csv"), index=False)

    # TXT summary → reviews folder
    os.makedirs("reviews", exist_ok=True)
    save_text_summary(df, title, tmdb_rating)

    # Replace chart
    if chart_canvas is not None:
        chart_canvas.get_tk_widget().destroy()
        chart_canvas = None

    fig = build_chart(df, title)
    chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    reset_btn()


def reset_btn():
    btn.config(state="normal", text="Analyze Movie")


# -----------------------------
# GUI SETUP
# -----------------------------
root = tk.Tk()
root.title("Movie Mood Analyzer")
root.geometry("1200x620")
root.configure(bg="#1e1e1e")

chart_canvas = None

tk.Label(root, text="🎬 Movie Mood Analyzer", font=("Arial", 22, "bold"),
         fg="white", bg="#1e1e1e").pack(pady=10)

search_frame = tk.Frame(root, bg="#1e1e1e")
search_frame.pack()

entry = tk.Entry(search_frame, font=("Arial", 14), width=36)
entry.pack(side="left", padx=(0, 8))
entry.insert(0, "e.g. The Shawshank Redemption")
entry.bind("<FocusIn>", lambda e: entry.delete(0, "end") if entry.get().startswith("e.g.") else None)

btn = tk.Button(search_frame, text="Analyze Movie", command=search_movie,
                bg="#4CAF50", fg="white", font=("Arial", 12), relief="flat", padx=10)
btn.pack(side="left")

status_label = tk.Label(root, text="Enter a movie name to get started",
                         fg="#aaaaaa", bg="#1e1e1e", font=("Arial", 10))
status_label.pack(pady=4)

content_frame = tk.Frame(root, bg="#1e1e1e")
content_frame.pack(fill="both", expand=True, padx=16, pady=8)

table_frame = tk.Frame(content_frame, bg="#1e1e1e")
table_frame.pack(side="left", fill="both", expand=True)

style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview", background="#2d2d2d", foreground="white",
                fieldbackground="#2d2d2d", rowheight=24, font=("Arial", 10))
style.configure("Treeview.Heading", background="#333333", foreground="white",
                font=("Arial", 11, "bold"))
style.map("Treeview", background=[("selected", "#4CAF50")])

tree = ttk.Treeview(table_frame,
                    columns=("Author", "Rating", "Sentiment", "Tokens", "Similarity", "Review"),
                    show="headings", height=18)

tree.heading("Author",     text="Author")
tree.heading("Rating",     text="Rating")
tree.heading("Sentiment",  text="Sentiment")
tree.heading("Tokens",     text="Tokens")
tree.heading("Similarity", text="Similarity")
tree.heading("Review",     text="Review Snippet")

tree.column("Author",     width=110, anchor="center")
tree.column("Rating",     width=65,  anchor="center")
tree.column("Sentiment",  width=85,  anchor="center")
tree.column("Tokens",     width=60,  anchor="center")
tree.column("Similarity", width=80,  anchor="center")
tree.column("Review",     width=380)

scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
tree.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

chart_frame = tk.Frame(content_frame, bg="#1e1e1e")
chart_frame.pack(side="right", fill="both", padx=(12, 0))

root.mainloop()