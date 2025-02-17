library(shiny)
library(shinythemes)
library(shinydashboard)
library(randomForest)



ui <- fluidPage(  theme = shinytheme("cerulean"),
                  
                  headerPanel(HTML("<h1 style='text-align: center; color:'BLUE'>MACHINE LEARNING FOR IRIS FLOWER</h1>")),
  sidebarLayout(
    sidebarPanel(
      numericInput("num", "Sepal Length", 5.1, min =0.1, max = 7.9, step = 0.1),
      numericInput("num1", "Sepal Width", 3.5, min =0.1, max = 4.4, step = 0.1),
      numericInput("num2", "Petal Length", 1.4, min =0.1, max = 6.9, step = 0.1),
      numericInput("num3", "Petal Width", 0.2, min =0.1, max = 2.5, step = 0.1),
      
      #accept csv file
      fileInput("file", "Upload CSV file"),
      
    actionButton("submit", "Submit")
    )
    
  
    ,
    
    
    
    mainPanel(
      # i want a text output here
      verbatimTextOutput("text"),
      
      
      # i want to display the output of the model here in table format
      tableOutput("table")
    )
    
    
    
    
  )
  
  
  
)


server <- function(input, output) {
  # upload model
  
model <- readRDS("model.rds")
  
  # output text should be displayed here saying welcome and predit
  output$text <- renderText({
    if(input$submit == 0){
      isolate("Welcome")
    } else {
      isolate("work done successfully")
    }
  
  })
  
  # output table should be displayed here
  output$table <- renderTable({
    if(input$submit == 0){
      return()
    } else {
      new_data <- data.frame(Sepal.Length = input$num, Sepal.Width = input$num1, Petal.Length = input$num2, Petal.Width = input$num3)
      prediction <- predict(model, new_data)
      fun <- function(x){
        if(x == "setosa"){
          return("Iris-setosa")
        } else if(x == "versicolor"){
          return("Iris-versicolor")
        } else {
          return("Iris-virginica")
        }
      }
    
      prediction <- sapply(prediction, fun)
      return(data.frame(new_data, prediction))
      
    }
  })
  
}

shinyApp(ui=ui, server=server)
