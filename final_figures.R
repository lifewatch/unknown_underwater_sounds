library(tidyverse)
library(forcats)
library(RColorBrewer)
library(rstatix)
library(ggsignif)
library(ggpubr)

#Author: Arienne.calonge@vliz.be

color = grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]

#set theme
theme_rbook <- function(base_size = 12, base_family = "Helvetica", base_line_size = base_size/12, 
                        base_rect_size = base_size/12){         
  theme( 
    axis.title = element_text(size = 12, family = "Helvetica"),                               
    axis.text= element_text(size = 12, family = "Helvetica"),                            
    plot.caption = element_text(size = 12, face = "italic", family = "Helvetica"),            
    panel.background = element_rect(fill="white"),                      
    axis.line = element_line(size = 1, colour = "black"),
    strip.background =element_rect(fill = "white"),
    panel.border = element_rect(colour = NA, fill=NA, size=0.5),
    strip.text = element_text(size=12, colour = "black", family = "Helvetica"),
    legend.key=element_blank(), 
    legend.title = element_text(size=12, family = "Helvetica"), 
    legend.text = element_text(size=12, family = "Helvetica"),
    #panel.grid.major.y = element_line(colour = "grey83"),
    #panel.grid.minor.y = element_line(colour = "grey83"),
    plot.background = element_blank()
  )
} 

#--------------count of labels
count_labels = AE_cropped %>% group_by(label) %>% summarize(count(label))


ggplot(data=AE_cropped,aes(fct_infreq(label)))+geom_histogram(stat="count",color="darkblue", fill="darkblue")+xlab("tag")+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+theme_rbook()

ggsave("plots_figures/annotations_count_raw.jpeg", dpi=300, width = 10, height = 5, device='jpeg') 

########------GRID SEARCH RESULTS------#######

results_grid_search = read_csv("plots_figures/results_grid_search.csv") %>% 
  filter(percentage_samples>0.15 & no_clusters <= 30) %>% 
  rowwise() %>% mutate(average_metrics = mean(c(homogeneity, DBCV), na.rm = TRUE))

#count samples per dataset
table(results_grid_search$`Feature description`)

#remove duplicate scores because of epsilon
results_grid_search = distinct(results_grid_search, homogeneity, DBCV, average_metrics, .keep_all= TRUE) 


bars <- gather(results_grid_search, Score, value, homogeneity:DBCV, factor_key=TRUE)
names(bars)[1]<- "Index"
names(bars)[3]<- "Feature_description"
bars <- bars %>%
  mutate(Feature_description = recode(Feature_description, "mean" = 'AVES-mean', 'max' = 'AVES-max', 'cropped' = 'CAE-crops', 'standard' = 'CAE-original'))

bars_homog <- bars %>% filter(Score == "homogeneity")
bars_DBCV <- bars %>% filter(Score == "DBCV")

bars = bars %>% filter(`Feature extraction`=="Aves")
ggplot(bars, aes(as.factor(Index))) + 
  geom_bar(stat="identity", position="dodge", aes(y=value, fill = Score))+scale_fill_brewer(palette="Paired")+
  geom_point(data = bars, aes(y = average_metrics), color="grey39") +
  geom_line(data = bars, aes(x = as.factor(Index), y = average_metrics, group = 1), color="grey39")+
  geom_hline(yintercept=max(bars$average_metrics), linetype="dashed", color = "red")+
  xlab("Grid search result index")+ylab("Score")+theme(axis.text.x = element_text(angle=90), legend.position="bottom")+
  facet_grid(. ~Feature_description, scales = "free_x", space = "free")+theme_rbook()

ggsave("plots_figures/scores_bar_chart_AVES.jpeg", dpi=300, width=12, height=5, device="jpeg")

#significant difference of means

stats = bars %>% group_by(Feature_description, Score) %>% 
  dplyr::summarise(n = n(), mean = mean(value, na.rm = TRUE), sd = sd(value, na.rm = TRUE))

write_csv(stats, "csv/stats.csv")

# grouped boxplot
my_colors <- RColorBrewer::brewer.pal(2, "Paired")[1:2]

ggplot(bars, aes(x=Score, y=value, fill=Feature_description)) + 
  geom_boxplot()+xlab("Scoring metric") + ylab("Score")+theme_rbook()+scale_fill_manual(values= c(my_colors, "gold", "goldenrod"))+
  guides(fill=guide_legend(title="Feature extraction"))+
  geom_signif(data = bars, aes(xmin = c(0.72), xmax = c(0.905), annotations = "***", y_position = c(0.85)),
              textsize = 4,vjust = 0.5, manual = TRUE)+
  geom_signif(data = bars, aes(xmin = c(1.1), xmax = c(1.3), annotations = "***", y_position = c(0.85)),
              textsize = 4,vjust = 0.5, manual = TRUE)+
  geom_signif(data = bars, aes(xmin = c(0.905), xmax = c(1.1), annotations = "***", y_position = c(0.85)),
              textsize = 4,vjust = 0.5, manual = TRUE)+
  geom_signif(data = bars, aes(xmin = c(0.905), xmax = c(1.3), annotations = "***", y_position = c(0.9)),
              textsize = 4,vjust = 0.5, manual = TRUE)+
  geom_signif(data = bars, aes(xmin = c(0.72), xmax = c(1.3), annotations = "***", y_position = c(0.95)),
              textsize = 4,vjust = 0.5, manual = TRUE)+
  
  geom_signif(data = bars, aes(xmin = c(1.72), xmax = c(1.9), annotations = "*", y_position = c(0.65)),
              textsize = 4,vjust = 0.5, manual = TRUE)+
  geom_signif(data = bars, aes(xmin = c(1.9), xmax = c(2.1), annotations = "*", y_position = c(0.7)),
              textsize = 4,vjust = 0.5, manual = TRUE)+
  geom_signif(data = bars, aes(xmin = c(1.72), xmax = c(2.1), annotations = "**", y_position = c(0.75)),
              textsize = 4,vjust = 0.5, manual = TRUE)

ggsave("plots_figures/scores_boxplot.jpeg", dpi=300, width=9, height=5, device="jpeg")

#check outliers
bars_homog %>% group_by(Feature_description) %>% identify_outliers(value)
bars_DBCV %>% group_by(Feature_description) %>% identify_outliers(value)
#remove outliers
bars_DBCV = bars_DBCV %>% filter(value > 0.17)
results_grid_search = results_grid_search %>% filter(DBCV > 0.17)

#test for normality
bars_homog %>% group_by(Feature_description) %>% shapiro_test(value)
bars_DBCV %>% group_by(Feature_description) %>% shapiro_test(value) #not normal - AE-standard

#equal variances
bars_homog %>% levene_test(value ~ Feature_description) # p =0.00000280, so reject Ho (unequal variances), therefore we cannot use ANOVA.
bars_DBCV %>% levene_test(value ~ Feature_description) # p = 0.00986

#use kruskal-wallis test
kruskal.test(value ~ Feature_description, data = bars_homog) #significant difference: Kruskal-Wallis chi-squared = 44.915, df = 3, p-value = 9.647e-10
kruskal.test(value ~ Feature_description, data = bars_DBCV)

#pairwise comparisons using Wilcoxon
pairwise.wilcox.test(bars_homog$value, bars_homog$Feature_description,p.adjust.method = "BH")
pairwise.wilcox.test(bars_DBCV$value, bars_DBCV$Feature_description,p.adjust.method = "BH")

#generalized linear models of variables
model_homog <- glm(homogeneity ~ (`Number of features`+min_cluster_size+min_samples)^2, family = Gamma(link="identity"), data = results_grid_search)
summary(model_homog)

model_DBCV <- glm(DBCV ~ (`Number of features`+min_cluster_size+min_samples)^2, family = Gamma(link="identity"), data = results_grid_search)
summary(model_DBCV)

#check assumptions for gamma glm
plot(model_DBCV,ask=FALSE)
plot(model_homog,ask=FALSE)

# Spearman correlation tests

cor.test(bars_homog$`Number of features`, bars_homog$value,  method = "spearman")
cor.test(bars_homog$min_cluster_size, bars_homog$value,  method = "spearman")
cor.test(bars_homog$min_samples, bars_homog$value,  method = "spearman")

cor.test(bars_DBCV$`Number of features`, bars_DBCV$value,  method = "spearman")
cor.test(bars_DBCV$min_cluster_size, bars_DBCV$value,  method = "spearman")
cor.test(bars_DBCV$min_samples, bars_DBCV$value,  method = "spearman")

########-----BEST GRID SEARCH------#######

best_grid_search = read_csv("plots_figures/366_results_grid_search.csv")
best_grid_search <- best_grid_search[2:24] %>% dplyr::mutate(ID=row_number()) %>% relocate(ID)
best_grid_search$cluster = best_grid_search$cluster+1

labels = best_grid_search %>% group_by(cluster, label) %>% summarise(n=n()) %>% add_count(cluster, wt = n) %>% mutate(percent = n/nn*100)

#----------------------------Confusion matrix

clust_df <- na.omit(best_grid_search)
clust_df$cluster = clust_df$cluster+1

confusion_df = as.data.frame(as.matrix(table(Labels = clust_df$label, Predicted_cluster = clust_df$cluster)))

confusion_df$Total = ave(confusion_df$Freq, confusion_df$Predicted_cluster, FUN=sum)

confusion_df= confusion_df %>% mutate(percentage = Freq/Total)

ggplot(confusion_df, aes(Predicted_cluster, Labels, fill= percentage)) + 
  geom_tile() + scale_fill_gradient(low="darkblue", high="yellow") +
  theme_minimal() + labs(fill = "Homogeneity")+xlab("Predicted cluster")

ggsave("plots_figures/heat_map.pdf", dpi=300, width=8, height=4, device="pdf")

#-----------------------------UMAP Projection
library(umap)

clust.data <- best_grid_search[1:21] 
clust.label <- best_grid_search[, c("cluster","ID", "label")]

set.seed(142)

clust_umap <- clust.data %>% column_to_rownames("ID") %>% umap()

umap_df <- clust_umap$layout %>% as.data.frame() %>% dplyr::rename(UMAP1 = "V1", UMAP2 = "V2") %>% 
  dplyr::mutate(ID= row_number()) %>% inner_join(clust.label, by = "ID")

umap_df$cluster = umap_df$cluster
umap_df$cluster <- as.factor(umap_df$cluster)

palette = (sample(color, 24))
umap_df %>% ggplot(aes(x=UMAP1, y=UMAP2, color = cluster))+ geom_point()+scale_color_manual(values =palette) + labs(x ="UMAP1", y="UMAP2")+theme_rbook()+theme(axis.text=element_blank())

ggsave("plots_figures/umap.jpg", dpi=300, width=6, height=5, device="jpeg")

