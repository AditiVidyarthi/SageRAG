"""User interface for the Research Assistant."""
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import os
from typing import Optional

from src.research_assistant import ResearchAssistant
from src.utils import load_models_and_db

console = Console()

class ResearchAssistantUI:
    """User interface for interacting with the Research Assistant."""
    
    def __init__(self, assistant: ResearchAssistant):
        """Initialize the UI with a research assistant instance."""
        self.assistant = assistant
        
    def display_welcome(self):
        """Display a welcome message."""
        console.print(Panel.fit(
            Markdown("# SageRAG Research Assistant\n\nAsk questions about scientific papers in the database."),
            title="Welcome",
            border_style="cyan"
        ))
        
    def run_cli(self):
        """Run the command line interface."""
        self.display_welcome()
        
        # Main interaction loop
        while True:
            console.print("\n[bold cyan]Options:[/bold cyan]")
            console.print("1. Ask a question")
            console.print("2. View conversation history")
            console.print("3. Clear conversation history")
            console.print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                query = input("\n[bold]Enter your query:[/bold] ")
                if not query.strip():
                    console.print("[yellow]Empty query. Please try again.[/yellow]")
                    continue
                    
                # Ask the user whether to use memory mode or full pipeline
                use_memory_input = input("Use conversation memory? (y/n): ").lower()
                use_memory = use_memory_input.startswith('y')
                
                # Process the query
                console.print("\n[bold cyan]Processing your query...[/bold cyan]")
                answer = self.assistant.process_query(query, use_memory=use_memory)
                
                # Display the final answer nicely formatted
                console.print(Panel(
                    Markdown(answer),
                    title="Answer",
                    border_style="green",
                    width=100
                ))
                
            elif choice == "2":
                history = self.assistant.get_chat_history()
                console.print(Panel(
                    history if history else "No conversation history.",
                    title="Conversation History",
                    border_style="blue"
                ))
                
            elif choice == "3":
                self.assistant.clear_memory()
                
            elif choice == "4":
                console.print("[bold green]Thank you for using SageRAG Research Assistant. Goodbye![/bold green]")
                break
                
            else:
                console.print("[yellow]Invalid choice. Please enter a number from 1-4.[/yellow]")

def main():
    """Main function to run the research assistant UI."""
    try:
        # Load models and database
        console.print("[bold cyan]Initializing SageRAG Research Assistant...[/bold cyan]")
        
        assistant = load_models_and_db()
        
        # Start the UI
        ui = ResearchAssistantUI(assistant)
        ui.run_cli()
        
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main()