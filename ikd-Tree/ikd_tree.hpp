/*
Description: ikd-Tree: an incremental k-d tree for robotic applications
Author: Yixi Cai
email: yixicai@connect.hku.hk
*/

#pragma once
#include <pthread.h>
#include <unistd.h>
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <queue>

namespace ikdTree {
#define EPSS 1e-6
#define Minimal_Unbalanced_Tree_Size 10
#define rebuildPointThreshold_ 1500
#define DOWNSAMPLE_SWITCH true
#define Q_LEN 1000000

using namespace std;

struct BoxPointType {
  double minVertex[3];
  double maxVertex[3];
};

enum Operation { ADD_POINT, DELETE_POINT, DELETE_BOX, ADD_BOX, DOWNSAMPLE_DELETE, PUSH_DOWN };

enum DeletedPointStorage { NOT_RECORD, DELETE_POINTS_REC, MULTI_THREAD_REC };

template <typename T>
class ManualQueue {
 private:
  int head_ = 0, tail_ = 0, counter_ = 0;
  T q_[Q_LEN];
  bool isEmpty = true;

 public:
  void pop() {
    if (counter_ == 0) return;
    head_++;
    head_ %= Q_LEN;
    counter_--;
    if (counter_ == 0) isEmpty = true;
  };
  T front() { return q_[head_]; };
  T back() { return q_[tail_]; };
  void clear() {
    head_ = 0;
    tail_ = 0;
    counter_ = 0;
    isEmpty = true;
  };
  void push(T op) {
    q_[tail_] = op;
    counter_++;
    if (isEmpty) isEmpty = false;
    tail_++;
    tail_ %= Q_LEN;
  };
  bool empty() { return isEmpty; };
  int size() { return counter_; };
};

template <typename Scalar>
class KdTree {
 public:
  using PointType = Eigen::Matrix<Scalar, 3, 1>;
  using PointVector = vector<PointType>;
  using Ptr = shared_ptr<KdTree<PointType>>;
  struct KdTreeNode {
    PointType point_;
    uint8_t divisionAxis_ = 0;
    int treeSize_ = 1;
    int invalidPointNum_ = 0;
    int downsampledDeleteNum_ = 0;
    bool pointDeleted_ = false;
    bool treeDeleted_ = false;
    bool pointDeletedDownsample_ = false;
    bool treeDeletedDownsample_ = false;
    bool needsPushDownLeft_ = false;
    bool needsPushDownRight_ = false;
    bool isWorking = false;
    double radiusSq_ = 0;
    pthread_mutex_t pushDownLock_{};
    double nodeRangeX_[2]{}, nodeRangeY_[2]{}, nodeRangeZ_[2]{};
    KdTreeNode* leftSon_ = nullptr;
    KdTreeNode* rightSon_ = nullptr;
    KdTreeNode* father_ = nullptr;
    // For paper data record
    double alphaDel_ = 0;
    double alphaBal_ = 0;
  };

  struct OperationLoggerType {
    PointType point_;
    BoxPointType boxpoint;
    bool treeDeleted_, treeDeletedDownsample_;
    Operation op;
  };

  struct PointType_CMP {
    PointType point_;
    double dist_ = 0.0;
    explicit PointType_CMP(PointType p = PointType(), double d = INFINITY) {
      this->point_ = p;
      this->dist_ = d;
    };
    bool operator<(const PointType_CMP& a) const {
      if (fabs(dist_ - a.dist_) < 1e-10)
        return point_.x() < a.point_.x();
      else
        return dist_ < a.dist_;
    }
  };

  class ManualHeap {
   public:
    explicit ManualHeap(int maxCapacity = 100) {
      cap_ = maxCapacity;
      heap_ = new PointType_CMP[maxCapacity];
      heapSize_ = 0;
    }

    ~ManualHeap() { delete[] heap_; }

    void pop() {
      if (heapSize_ == 0) return;
      heap_[0] = heap_[heapSize_ - 1];
      heapSize_--;
      MoveDown(0);
    }

    PointType_CMP top() { return heap_[0]; }

    void push(PointType_CMP point_) {
      if (heapSize_ >= cap_) return;
      heap_[heapSize_] = point_;
      FloatUp(heapSize_);
      heapSize_++;
    }

    int size() { return heapSize_; }

    void clear() { heapSize_ = 0; }

   private:
    int heapSize_ = 0;
    int cap_ = 0;
    PointType_CMP* heap_;
    void MoveDown(int heapIndex) {
      int l = heapIndex * 2 + 1;
      PointType_CMP tmp = heap_[heapIndex];
      while (l < heapSize_) {
        if (l + 1 < heapSize_ && heap_[l] < heap_[l + 1]) l++;
        if (tmp < heap_[l]) {
          heap_[heapIndex] = heap_[l];
          heapIndex = l;
          l = heapIndex * 2 + 1;
        } else
          break;
      }
      heap_[heapIndex] = tmp;
    }

    void FloatUp(int heapIndex) {
      int ancestor = (heapIndex - 1) / 2;
      PointType_CMP tmp = heap_[heapIndex];
      while (heapIndex > 0) {
        if (heap_[ancestor] < tmp) {
          heap_[heapIndex] = heap_[ancestor];
          heapIndex = ancestor;
          ancestor = (heapIndex - 1) / 2;
        } else
          break;
      }
      heap_[heapIndex] = tmp;
    }
  };

 private:
  // Multi-thread Tree rebuild
  bool isDone_ = false;
  bool isRebuild_ = false;
  pthread_t rebuildThread_;
  pthread_mutex_t isDoneLock_, rebuildPtrLock_, isRebuildLock_, searchLock_, rebuildLoggerLock_, deletePointsCacheLock_;
  ManualQueue<OperationLoggerType> rebuildLogger_;
  PointVector rebuildPtsStorage_;
  KdTreeNode** rebuildNode_ = nullptr;
  int searchLockCounter_ = 0;
  // KD Tree Functions and augmented variables
  int tmpTreeSize_ = 0;
  double tmpAlphaBal_ = 0.5, tmpAlphaDel_ = 0.0;
  double deleteCriterion_ = 0.5f;
  double balanceCriterion_ = 0.7f;
  double downsampleSize_ = 0.2f;
  KdTreeNode* STATIC_ROOT_NODE = nullptr;
  PointVector deletedPoints_;
  PointVector downsampledPoints_;
  PointVector deletedPointsMultithread_;

  static void* pthreadMultiThread(void* arg) {
    ((KdTree*)arg)->multiThreadRebuild();
    return nullptr;
  };

  void multiThreadRebuild() {
    bool terminated = false;
    KdTreeNode *father, **newNode;
    pthread_mutex_lock(&isDoneLock_);
    terminated = isDone_;
    pthread_mutex_unlock(&isDoneLock_);
    while (!terminated) {
      pthread_mutex_lock(&rebuildPtrLock_);
      pthread_mutex_lock(&isRebuildLock_);
      if (rebuildNode_ != nullptr) {
        /* Traverse and copy */
        if (!rebuildLogger_.empty()) {
          printf("\n\n\n\n\n\n\n\n\n\n\n ERROR!!! \n\n\n\n\n\n\n\n\n");
        }
        isRebuild_ = true;
        if (*rebuildNode_ == rootNode_) {
          tmpTreeSize_ = rootNode_->treeSize_;
          tmpAlphaBal_ = rootNode_->alphaBal_;
          tmpAlphaDel_ = rootNode_->alphaDel_;
        }
        KdTreeNode* oldRootNode = (*rebuildNode_);
        father = (*rebuildNode_)->father_;
        PointVector().swap(rebuildPtsStorage_);
        // Lock search
        pthread_mutex_lock(&searchLock_);
        while (searchLockCounter_ != 0) {
          pthread_mutex_unlock(&searchLock_);
          usleep(1);
          pthread_mutex_lock(&searchLock_);
        }
        searchLockCounter_ = -1;
        pthread_mutex_unlock(&searchLock_);
        // Lock deleted points cache
        pthread_mutex_lock(&deletePointsCacheLock_);
        flatten(*rebuildNode_, rebuildPtsStorage_, MULTI_THREAD_REC);
        // Unlock deleted points cache
        pthread_mutex_unlock(&deletePointsCacheLock_);
        // Unlock search
        pthread_mutex_lock(&searchLock_);
        searchLockCounter_ = 0;
        pthread_mutex_unlock(&searchLock_);
        pthread_mutex_unlock(&isRebuildLock_);
        /* rebuild and update missed operations*/
        OperationLoggerType Operation;
        KdTreeNode* newRootNode = nullptr;
        if (int(rebuildPtsStorage_.size()) > 0) {
          buildTree(&newRootNode, 0, rebuildPtsStorage_.size() - 1, rebuildPtsStorage_);
          // rebuild has been done. Updates the blocked operations into the new tree
          pthread_mutex_lock(&isRebuildLock_);
          pthread_mutex_lock(&rebuildLoggerLock_);
          int counter = 0;
          while (!rebuildLogger_.empty()) {
            Operation = rebuildLogger_.front();
            maxQueueSize_ = max(maxQueueSize_, rebuildLogger_.size());
            rebuildLogger_.pop();
            pthread_mutex_unlock(&rebuildLoggerLock_);
            pthread_mutex_unlock(&isRebuildLock_);
            runOperation(&newRootNode, Operation);
            counter++;
            if (counter % 10 == 0) usleep(1);
            pthread_mutex_lock(&isRebuildLock_);
            pthread_mutex_lock(&rebuildLoggerLock_);
          }
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        /* Replace to original tree*/
        // pthread_mutex_lock(&isRebuildLock_);
        pthread_mutex_lock(&searchLock_);
        while (searchLockCounter_ != 0) {
          pthread_mutex_unlock(&searchLock_);
          usleep(1);
          pthread_mutex_lock(&searchLock_);
        }
        searchLockCounter_ = -1;
        pthread_mutex_unlock(&searchLock_);
        if (father->leftSon_ == *rebuildNode_) {
          father->leftSon_ = newRootNode;
        } else if (father->rightSon_ == *rebuildNode_) {
          father->rightSon_ = newRootNode;
        } else {
          throw std::runtime_error("Error: Father ptr incompatible with current node\n");
        }
        if (newRootNode != nullptr) newRootNode->father_ = father;
        (*rebuildNode_) = newRootNode;
        int oldValid = oldRootNode->treeSize_ - oldRootNode->invalidPointNum_;
        int newValid = newRootNode->treeSize_ - newRootNode->invalidPointNum_;
        if (father == STATIC_ROOT_NODE) rootNode_ = STATIC_ROOT_NODE->leftSon_;
        KdTreeNode* updateRoot = *rebuildNode_;
        while (updateRoot != nullptr && updateRoot != rootNode_) {
          updateRoot = updateRoot->father_;
          if (updateRoot->isWorking) break;
          if (updateRoot == updateRoot->father_->leftSon_ && updateRoot->father_->needsPushDownLeft_) break;
          if (updateRoot == updateRoot->father_->rightSon_ && updateRoot->father_->needsPushDownRight_) break;
          update(updateRoot);
        }
        pthread_mutex_lock(&searchLock_);
        searchLockCounter_ = 0;
        pthread_mutex_unlock(&searchLock_);
        rebuildNode_ = nullptr;
        pthread_mutex_unlock(&isRebuildLock_);
        isRebuild_ = false;
        /* Delete discarded tree nodes */
        deleteTreeNodes(&oldRootNode);
      } else {
        pthread_mutex_unlock(&isRebuildLock_);
      }
      pthread_mutex_unlock(&rebuildPtrLock_);
      pthread_mutex_lock(&isDoneLock_);
      terminated = isDone_;
      pthread_mutex_unlock(&isDoneLock_);
      usleep(100);
    }
    printf("rebuild thread terminated normally\n");
  };
  void startThread() {
    pthread_mutex_init(&isDoneLock_, nullptr);
    pthread_mutex_init(&rebuildPtrLock_, nullptr);
    pthread_mutex_init(&rebuildLoggerLock_, nullptr);
    pthread_mutex_init(&deletePointsCacheLock_, nullptr);
    pthread_mutex_init(&isRebuildLock_, nullptr);
    pthread_mutex_init(&searchLock_, nullptr);
    pthread_create(&rebuildThread_, nullptr, pthreadMultiThread, this);
    printf("Multi thread started \n");
  };
  void stopThread() {
    pthread_mutex_lock(&isDoneLock_);
    isDone_ = true;
    pthread_mutex_unlock(&isDoneLock_);
    if (rebuildThread_) pthread_join(rebuildThread_, nullptr);
    pthread_mutex_destroy(&isDoneLock_);
    pthread_mutex_destroy(&rebuildLoggerLock_);
    pthread_mutex_destroy(&rebuildPtrLock_);
    pthread_mutex_destroy(&deletePointsCacheLock_);
    pthread_mutex_destroy(&isRebuildLock_);
    pthread_mutex_destroy(&searchLock_);
  };
  void runOperation(KdTreeNode** root, OperationLoggerType operation) {
    switch (operation.op) {
      case ADD_POINT:
        addPoint(root, operation.point_, false, (*root)->divisionAxis_);
        break;
      case ADD_BOX:
        addRange(root, operation.boxpoint, false);
        break;
      case DELETE_POINT:
        deletePoint(root, operation.point_, false);
        break;
      case DELETE_BOX:
        deleteRange(root, operation.boxpoint, false, false);
        break;
      case DOWNSAMPLE_DELETE:
        deleteRange(root, operation.boxpoint, false, true);
        break;
      case PUSH_DOWN:
        (*root)->treeDeletedDownsample_ |= operation.treeDeletedDownsample_;
        (*root)->pointDeletedDownsample_ |= operation.treeDeletedDownsample_;
        (*root)->treeDeleted_ = operation.treeDeleted_ || (*root)->treeDeletedDownsample_;
        (*root)->pointDeleted_ = (*root)->treeDeleted_ || (*root)->pointDeletedDownsample_;
        if (operation.treeDeletedDownsample_) (*root)->downsampledDeleteNum_ = (*root)->treeSize_;
        if (operation.treeDeleted_)
          (*root)->invalidPointNum_ = (*root)->treeSize_;
        else
          (*root)->invalidPointNum_ = (*root)->downsampledDeleteNum_;
        (*root)->needsPushDownLeft_ = true;
        (*root)->needsPushDownRight_ = true;
        break;
      default:
        break;
    }
  };

  void initializeTreeNode(KdTreeNode* root) {
    root->point_.x() = 0.0f;
    root->point_.y() = 0.0f;
    root->point_.z() = 0.0f;
    root->nodeRangeX_[0] = 0.0f;
    root->nodeRangeX_[1] = 0.0f;
    root->nodeRangeY_[0] = 0.0f;
    root->nodeRangeY_[1] = 0.0f;
    root->nodeRangeZ_[0] = 0.0f;
    root->nodeRangeZ_[1] = 0.0f;
    root->divisionAxis_ = 0;
    root->father_ = nullptr;
    root->leftSon_ = nullptr;
    root->rightSon_ = nullptr;
    root->treeSize_ = 0;
    root->invalidPointNum_ = 0;
    root->downsampledDeleteNum_ = 0;
    root->pointDeleted_ = false;
    root->treeDeleted_ = false;
    root->needsPushDownLeft_ = false;
    root->needsPushDownRight_ = false;
    root->pointDeletedDownsample_ = false;
    root->isWorking = false;
    pthread_mutex_init(&(root->pushDownLock_), nullptr);
  };

  void buildTree(KdTreeNode** root, int l, int r, PointVector& storage) {
    if (l > r) return;
    *root = new KdTreeNode;
    initializeTreeNode(*root);
    int mid = (l + r) >> 1;
    int divisionAxis = 0;
    int i;
    // Find the best division Axis
    double minValue[3] = {INFINITY, INFINITY, INFINITY};
    double maxValue[3] = {-INFINITY, -INFINITY, -INFINITY};
    double dimensionRange[3] = {0, 0, 0};
    for (i = l; i <= r; i++) {
      minValue[0] = min(minValue[0], storage[i].x());
      minValue[1] = min(minValue[1], storage[i].y());
      minValue[2] = min(minValue[2], storage[i].z());
      maxValue[0] = max(maxValue[0], storage[i].x());
      maxValue[1] = max(maxValue[1], storage[i].y());
      maxValue[2] = max(maxValue[2], storage[i].z());
    }
    // Select the longest dimension as division axis
    for (i = 0; i < 3; i++) dimensionRange[i] = maxValue[i] - minValue[i];
    for (i = 1; i < 3; i++)
      if (dimensionRange[i] > dimensionRange[divisionAxis]) divisionAxis = i;
    // Divide by the division axis and recursively build.

    (*root)->divisionAxis_ = divisionAxis;
    switch (divisionAxis) {
      case 0:
        nth_element(begin(storage) + l, begin(storage) + mid, begin(storage) + r + 1, compareX);
        break;
      case 1:
        nth_element(begin(storage) + l, begin(storage) + mid, begin(storage) + r + 1, compareY);
        break;
      case 2:
        nth_element(begin(storage) + l, begin(storage) + mid, begin(storage) + r + 1, compareZ);
        break;
      default:
        nth_element(begin(storage) + l, begin(storage) + mid, begin(storage) + r + 1, compareX);
        break;
    }
    (*root)->point_ = storage[mid];
    KdTreeNode *leftSon = nullptr, *rightSon = nullptr;
    buildTree(&leftSon, l, mid - 1, storage);
    buildTree(&rightSon, mid + 1, r, storage);
    (*root)->leftSon_ = leftSon;
    (*root)->rightSon_ = rightSon;
    update((*root));
  };
  void rebuild(KdTreeNode** root) {
    KdTreeNode* father;
    if ((*root)->treeSize_ >= rebuildPointThreshold_) {
      if (!pthread_mutex_trylock(&rebuildPtrLock_)) {
        if (rebuildNode_ == nullptr || ((*root)->treeSize_ > (*rebuildNode_)->treeSize_)) {
          rebuildNode_ = root;
        }
        pthread_mutex_unlock(&rebuildPtrLock_);
      }
    } else {
      father = (*root)->father_;
      int treeSize = (*root)->treeSize_;
      pointStorage_.clear();
      flatten(*root, pointStorage_, DELETE_POINTS_REC);
      deleteTreeNodes(root);
      buildTree(root, 0, pointStorage_.size() - 1, pointStorage_);
      if (*root != nullptr) (*root)->father_ = father;
      if (*root == rootNode_) STATIC_ROOT_NODE->leftSon_ = *root;
    }
  };
  int deleteRange(KdTreeNode** root, BoxPointType boxpoint, bool allowRebuild, bool shouldDownsample) {
    if ((*root) == nullptr || (*root)->treeDeleted_) return 0;
    (*root)->isWorking = true;
    pushDown(*root);
    int counter = 0;
    if (boxpoint.maxVertex[0] <= (*root)->nodeRangeX_[0] || boxpoint.minVertex[0] > (*root)->nodeRangeX_[1]) return 0;
    if (boxpoint.maxVertex[1] <= (*root)->nodeRangeY_[0] || boxpoint.minVertex[1] > (*root)->nodeRangeY_[1]) return 0;
    if (boxpoint.maxVertex[2] <= (*root)->nodeRangeZ_[0] || boxpoint.minVertex[2] > (*root)->nodeRangeZ_[1]) return 0;
    if (boxpoint.minVertex[0] <= (*root)->nodeRangeX_[0] && boxpoint.maxVertex[0] > (*root)->nodeRangeX_[1] &&
        boxpoint.minVertex[1] <= (*root)->nodeRangeY_[0] && boxpoint.maxVertex[1] > (*root)->nodeRangeY_[1] &&
        boxpoint.minVertex[2] <= (*root)->nodeRangeZ_[0] && boxpoint.maxVertex[2] > (*root)->nodeRangeZ_[1]) {
      (*root)->treeDeleted_ = true;
      (*root)->pointDeleted_ = true;
      (*root)->needsPushDownLeft_ = true;
      (*root)->needsPushDownRight_ = true;
      counter = (*root)->treeSize_ - (*root)->invalidPointNum_;
      (*root)->invalidPointNum_ = (*root)->treeSize_;
      if (shouldDownsample) {
        (*root)->treeDeletedDownsample_ = true;
        (*root)->pointDeletedDownsample_ = true;
        (*root)->downsampledDeleteNum_ = (*root)->treeSize_;
      }
      return counter;
    }
    if (!(*root)->pointDeleted_ && boxpoint.minVertex[0] <= (*root)->point_.x() && boxpoint.maxVertex[0] > (*root)->point_.x() &&
        boxpoint.minVertex[1] <= (*root)->point_.y() && boxpoint.maxVertex[1] > (*root)->point_.y() &&
        boxpoint.minVertex[2] <= (*root)->point_.z() && boxpoint.maxVertex[2] > (*root)->point_.z()) {
      (*root)->pointDeleted_ = true;
      counter += 1;
      if (shouldDownsample) (*root)->pointDeletedDownsample_ = true;
    }
    OperationLoggerType boxDeleteOperation;
    if (shouldDownsample)
      boxDeleteOperation.op = DOWNSAMPLE_DELETE;
    else
      boxDeleteOperation.op = DELETE_BOX;
    boxDeleteOperation.boxpoint = boxpoint;
    if ((rebuildNode_ == nullptr) || (*root)->leftSon_ != *rebuildNode_) {
      counter += deleteRange(&((*root)->leftSon_), boxpoint, allowRebuild, shouldDownsample);
    } else {
      pthread_mutex_lock(&isRebuildLock_);
      counter += deleteRange(&((*root)->leftSon_), boxpoint, false, shouldDownsample);
      if (isRebuild_) {
        pthread_mutex_lock(&rebuildLoggerLock_);
        rebuildLogger_.push(boxDeleteOperation);
        pthread_mutex_unlock(&rebuildLoggerLock_);
      }
      pthread_mutex_unlock(&isRebuildLock_);
    }
    if ((rebuildNode_ == nullptr) || (*root)->rightSon_ != *rebuildNode_) {
      counter += deleteRange(&((*root)->rightSon_), boxpoint, allowRebuild, shouldDownsample);
    } else {
      pthread_mutex_lock(&isRebuildLock_);
      counter += deleteRange(&((*root)->rightSon_), boxpoint, false, shouldDownsample);
      if (isRebuild_) {
        pthread_mutex_lock(&rebuildLoggerLock_);
        rebuildLogger_.push(boxDeleteOperation);
        pthread_mutex_unlock(&rebuildLoggerLock_);
      }
      pthread_mutex_unlock(&isRebuildLock_);
    }
    update(*root);
    if (rebuildNode_ != nullptr && *rebuildNode_ == *root && (*root)->treeSize_ < rebuildPointThreshold_) rebuildNode_ = nullptr;
    bool needsRebuild = allowRebuild & criterionCheck((*root));
    if (needsRebuild) rebuild(root);
    if ((*root) != nullptr) (*root)->isWorking = false;
    return counter;
  };

  void deletePoint(KdTreeNode** root, PointType point, bool allowRebuild) {
    if ((*root) == nullptr || (*root)->treeDeleted_) return;
    (*root)->isWorking = true;
    pushDown(*root);
    if (isSamePoint((*root)->point_, point) && !(*root)->pointDeleted_) {
      (*root)->pointDeleted_ = true;
      (*root)->invalidPointNum_ += 1;
      if ((*root)->invalidPointNum_ == (*root)->treeSize_) (*root)->treeDeleted_ = true;
      return;
    }
    OperationLoggerType deleteOperation;
    deleteOperation.op = DELETE_POINT;
    deleteOperation.point_ = point;
    if (((*root)->divisionAxis_ == 0 && point.x() < (*root)->point_.x()) ||
        ((*root)->divisionAxis_ == 1 && point.y() < (*root)->point_.y()) ||
        ((*root)->divisionAxis_ == 2 && point.z() < (*root)->point_.z())) {
      if ((rebuildNode_ == nullptr) || (*root)->leftSon_ != *rebuildNode_) {
        deletePoint(&(*root)->leftSon_, point, allowRebuild);
      } else {
        pthread_mutex_lock(&isRebuildLock_);
        deletePoint(&(*root)->leftSon_, point, false);
        if (isRebuild_) {
          pthread_mutex_lock(&rebuildLoggerLock_);
          rebuildLogger_.push(deleteOperation);
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        pthread_mutex_unlock(&isRebuildLock_);
      }
    } else {
      if ((rebuildNode_ == nullptr) || (*root)->rightSon_ != *rebuildNode_) {
        deletePoint(&(*root)->rightSon_, point, allowRebuild);
      } else {
        pthread_mutex_lock(&isRebuildLock_);
        deletePoint(&(*root)->rightSon_, point, false);
        if (isRebuild_) {
          pthread_mutex_lock(&rebuildLoggerLock_);
          rebuildLogger_.push(deleteOperation);
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        pthread_mutex_unlock(&isRebuildLock_);
      }
    }
    update(*root);
    if (rebuildNode_ != nullptr && *rebuildNode_ == *root && (*root)->treeSize_ < rebuildPointThreshold_) rebuildNode_ = nullptr;
    bool needsRebuild = allowRebuild & criterionCheck((*root));
    if (needsRebuild) rebuild(root);
    if ((*root) != nullptr) (*root)->isWorking = false;
  };

  void addPoint(KdTreeNode** root, PointType point_, bool allowRebuild, int father_axis) {
    if (*root == nullptr) {
      *root = new KdTreeNode;
      initializeTreeNode(*root);
      (*root)->point_ = point_;
      (*root)->divisionAxis_ = (father_axis + 1) % 3;
      update(*root);
      return;
    }
    (*root)->isWorking = true;
    OperationLoggerType addOperation;
    addOperation.op = ADD_POINT;
    addOperation.point_ = point_;
    pushDown(*root);
    if (((*root)->divisionAxis_ == 0 && point_.x() < (*root)->point_.x()) ||
        ((*root)->divisionAxis_ == 1 && point_.y() < (*root)->point_.y()) ||
        ((*root)->divisionAxis_ == 2 && point_.z() < (*root)->point_.z())) {
      if ((rebuildNode_ == nullptr) || (*root)->leftSon_ != *rebuildNode_) {
        addPoint(&(*root)->leftSon_, point_, allowRebuild, (*root)->divisionAxis_);
      } else {
        pthread_mutex_lock(&isRebuildLock_);
        addPoint(&(*root)->leftSon_, point_, false, (*root)->divisionAxis_);
        if (isRebuild_) {
          pthread_mutex_lock(&rebuildLoggerLock_);
          rebuildLogger_.push(addOperation);
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        pthread_mutex_unlock(&isRebuildLock_);
      }
    } else {
      if ((rebuildNode_ == nullptr) || (*root)->rightSon_ != *rebuildNode_) {
        addPoint(&(*root)->rightSon_, point_, allowRebuild, (*root)->divisionAxis_);
      } else {
        pthread_mutex_lock(&isRebuildLock_);
        addPoint(&(*root)->rightSon_, point_, false, (*root)->divisionAxis_);
        if (isRebuild_) {
          pthread_mutex_lock(&rebuildLoggerLock_);
          rebuildLogger_.push(addOperation);
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        pthread_mutex_unlock(&isRebuildLock_);
      }
    }
    update(*root);
    if (rebuildNode_ != nullptr && *rebuildNode_ == *root && (*root)->treeSize_ < rebuildPointThreshold_) rebuildNode_ = nullptr;
    bool needsRebuild = allowRebuild & criterionCheck((*root));
    if (needsRebuild) rebuild(root);
    if ((*root) != nullptr) (*root)->isWorking = false;
  };

  void addRange(KdTreeNode** root, BoxPointType boxpoint, bool allowRebuild) {
    if ((*root) == nullptr) return;
    (*root)->isWorking = true;
    pushDown(*root);
    if (boxpoint.maxVertex[0] <= (*root)->nodeRangeX_[0] || boxpoint.minVertex[0] > (*root)->nodeRangeX_[1]) return;
    if (boxpoint.maxVertex[1] <= (*root)->nodeRangeY_[0] || boxpoint.minVertex[1] > (*root)->nodeRangeY_[1]) return;
    if (boxpoint.maxVertex[2] <= (*root)->nodeRangeZ_[0] || boxpoint.minVertex[2] > (*root)->nodeRangeZ_[1]) return;
    if (boxpoint.minVertex[0] <= (*root)->nodeRangeX_[0] && boxpoint.maxVertex[0] > (*root)->nodeRangeX_[1] &&
        boxpoint.minVertex[1] <= (*root)->nodeRangeY_[0] && boxpoint.maxVertex[1] > (*root)->nodeRangeY_[1] &&
        boxpoint.minVertex[2] <= (*root)->nodeRangeZ_[0] && boxpoint.maxVertex[2] > (*root)->nodeRangeZ_[1]) {
      (*root)->treeDeleted_ = (*root)->treeDeletedDownsample_;
      (*root)->pointDeleted_ = (*root)->pointDeletedDownsample_;
      (*root)->needsPushDownLeft_ = true;
      (*root)->needsPushDownRight_ = true;
      (*root)->invalidPointNum_ = (*root)->downsampledDeleteNum_;
      return;
    }
    if (boxpoint.minVertex[0] <= (*root)->point_.x() && boxpoint.maxVertex[0] > (*root)->point_.x() &&
        boxpoint.minVertex[1] <= (*root)->point_.y() && boxpoint.maxVertex[1] > (*root)->point_.y() &&
        boxpoint.minVertex[2] <= (*root)->point_.z() && boxpoint.maxVertex[2] > (*root)->point_.z()) {
      (*root)->pointDeleted_ = (*root)->pointDeletedDownsample_;
    }
    OperationLoggerType boxAddOperation;
    boxAddOperation.op = ADD_BOX;
    boxAddOperation.boxpoint = boxpoint;
    if ((rebuildNode_ == nullptr) || (*root)->leftSon_ != *rebuildNode_) {
      addRange(&((*root)->leftSon_), boxpoint, allowRebuild);
    } else {
      pthread_mutex_lock(&isRebuildLock_);
      addRange(&((*root)->leftSon_), boxpoint, false);
      if (isRebuild_) {
        pthread_mutex_lock(&rebuildLoggerLock_);
        rebuildLogger_.push(boxAddOperation);
        pthread_mutex_unlock(&rebuildLoggerLock_);
      }
      pthread_mutex_unlock(&isRebuildLock_);
    }
    if ((rebuildNode_ == nullptr) || (*root)->rightSon_ != *rebuildNode_) {
      addRange(&((*root)->rightSon_), boxpoint, allowRebuild);
    } else {
      pthread_mutex_lock(&isRebuildLock_);
      addRange(&((*root)->rightSon_), boxpoint, false);
      if (isRebuild_) {
        pthread_mutex_lock(&rebuildLoggerLock_);
        rebuildLogger_.push(boxAddOperation);
        pthread_mutex_unlock(&rebuildLoggerLock_);
      }
      pthread_mutex_unlock(&isRebuildLock_);
    }
    update(*root);
    if (rebuildNode_ != nullptr && *rebuildNode_ == *root && (*root)->treeSize_ < rebuildPointThreshold_) rebuildNode_ = nullptr;
    bool needsRebuild = allowRebuild & criterionCheck((*root));
    if (needsRebuild) rebuild(root);
    if ((*root) != nullptr) (*root)->isWorking = false;
  };

  void search(KdTreeNode* root, int kNearest, PointType point, ManualHeap& q, double maxDistance) {
    if (root == nullptr || root->treeDeleted_) return;
    double currentDistance = calculateBoxDistance(root, point);
    double maxDistanceSquared = maxDistance * maxDistance;
    if (currentDistance > maxDistanceSquared) return;
    int retval;
    if (root->needsPushDownLeft_ || root->needsPushDownRight_) {
      retval = pthread_mutex_trylock(&(root->pushDownLock_));
      if (retval == 0) {
        pushDown(root);
        pthread_mutex_unlock(&(root->pushDownLock_));
      } else {
        pthread_mutex_lock(&(root->pushDownLock_));
        pthread_mutex_unlock(&(root->pushDownLock_));
      }
    }
    if (!root->pointDeleted_) {
      double dist = calculateDistance(point, root->point_);
      if (dist <= maxDistanceSquared && (q.size() < kNearest || dist < q.top().dist_)) {
        if (q.size() >= kNearest) q.pop();
        PointType_CMP currentPoint{root->point_, dist};
        q.push(currentPoint);
      }
    }
    int cur_search_counter;
    double leftNodeDistance = calculateBoxDistance(root->leftSon_, point);
    double rightNodeDistance = calculateBoxDistance(root->rightSon_, point);
    if (q.size() < kNearest || leftNodeDistance < q.top().dist_ && rightNodeDistance < q.top().dist_) {
      if (leftNodeDistance <= rightNodeDistance) {
        if (rebuildNode_ == nullptr || *rebuildNode_ != root->leftSon_) {
          search(root->leftSon_, kNearest, point, q, maxDistance);
        } else {
          pthread_mutex_lock(&searchLock_);
          while (searchLockCounter_ == -1) {
            pthread_mutex_unlock(&searchLock_);
            usleep(1);
            pthread_mutex_lock(&searchLock_);
          }
          searchLockCounter_ += 1;
          pthread_mutex_unlock(&searchLock_);
          search(root->leftSon_, kNearest, point, q, maxDistance);
          pthread_mutex_lock(&searchLock_);
          searchLockCounter_ -= 1;
          pthread_mutex_unlock(&searchLock_);
        }
        if (q.size() < kNearest || rightNodeDistance < q.top().dist_) {
          if (rebuildNode_ == nullptr || *rebuildNode_ != root->rightSon_) {
            search(root->rightSon_, kNearest, point, q, maxDistance);
          } else {
            pthread_mutex_lock(&searchLock_);
            while (searchLockCounter_ == -1) {
              pthread_mutex_unlock(&searchLock_);
              usleep(1);
              pthread_mutex_lock(&searchLock_);
            }
            searchLockCounter_ += 1;
            pthread_mutex_unlock(&searchLock_);
            search(root->rightSon_, kNearest, point, q, maxDistance);
            pthread_mutex_lock(&searchLock_);
            searchLockCounter_ -= 1;
            pthread_mutex_unlock(&searchLock_);
          }
        }
      } else {
        if (rebuildNode_ == nullptr || *rebuildNode_ != root->rightSon_) {
          search(root->rightSon_, kNearest, point, q, maxDistance);
        } else {
          pthread_mutex_lock(&searchLock_);
          while (searchLockCounter_ == -1) {
            pthread_mutex_unlock(&searchLock_);
            usleep(1);
            pthread_mutex_lock(&searchLock_);
          }
          searchLockCounter_ += 1;
          pthread_mutex_unlock(&searchLock_);
          search(root->rightSon_, kNearest, point, q, maxDistance);
          pthread_mutex_lock(&searchLock_);
          searchLockCounter_ -= 1;
          pthread_mutex_unlock(&searchLock_);
        }
        if (q.size() < kNearest || leftNodeDistance < q.top().dist_) {
          if (rebuildNode_ == nullptr || *rebuildNode_ != root->leftSon_) {
            search(root->leftSon_, kNearest, point, q, maxDistance);
          } else {
            pthread_mutex_lock(&searchLock_);
            while (searchLockCounter_ == -1) {
              pthread_mutex_unlock(&searchLock_);
              usleep(1);
              pthread_mutex_lock(&searchLock_);
            }
            searchLockCounter_ += 1;
            pthread_mutex_unlock(&searchLock_);
            search(root->leftSon_, kNearest, point, q, maxDistance);
            pthread_mutex_lock(&searchLock_);
            searchLockCounter_ -= 1;
            pthread_mutex_unlock(&searchLock_);
          }
        }
      }
    } else {
      if (leftNodeDistance < q.top().dist_) {
        if (rebuildNode_ == nullptr || *rebuildNode_ != root->leftSon_) {
          search(root->leftSon_, kNearest, point, q, maxDistance);
        } else {
          pthread_mutex_lock(&searchLock_);
          while (searchLockCounter_ == -1) {
            pthread_mutex_unlock(&searchLock_);
            usleep(1);
            pthread_mutex_lock(&searchLock_);
          }
          searchLockCounter_ += 1;
          pthread_mutex_unlock(&searchLock_);
          search(root->leftSon_, kNearest, point, q, maxDistance);
          pthread_mutex_lock(&searchLock_);
          searchLockCounter_ -= 1;
          pthread_mutex_unlock(&searchLock_);
        }
      }
      if (rightNodeDistance < q.top().dist_) {
        if (rebuildNode_ == nullptr || *rebuildNode_ != root->rightSon_) {
          search(root->rightSon_, kNearest, point, q, maxDistance);
        } else {
          pthread_mutex_lock(&searchLock_);
          while (searchLockCounter_ == -1) {
            pthread_mutex_unlock(&searchLock_);
            usleep(1);
            pthread_mutex_lock(&searchLock_);
          }
          searchLockCounter_ += 1;
          pthread_mutex_unlock(&searchLock_);
          search(root->rightSon_, kNearest, point, q, maxDistance);
          pthread_mutex_lock(&searchLock_);
          searchLockCounter_ -= 1;
          pthread_mutex_unlock(&searchLock_);
        }
      }
    }
  };

  void searchRange(KdTreeNode* root, BoxPointType boxpoint, PointVector& storage) {
    if (root == nullptr) return;
    pushDown(root);
    if (boxpoint.maxVertex[0] <= root->nodeRangeX_[0] || boxpoint.minVertex[0] > root->nodeRangeX_[1]) return;
    if (boxpoint.maxVertex[1] <= root->nodeRangeY_[0] || boxpoint.minVertex[1] > root->nodeRangeY_[1]) return;
    if (boxpoint.maxVertex[2] <= root->nodeRangeZ_[0] || boxpoint.minVertex[2] > root->nodeRangeZ_[1]) return;
    if (boxpoint.minVertex[0] <= root->nodeRangeX_[0] && boxpoint.maxVertex[0] > root->nodeRangeX_[1] &&
        boxpoint.minVertex[1] <= root->nodeRangeY_[0] && boxpoint.maxVertex[1] > root->nodeRangeY_[1] &&
        boxpoint.minVertex[2] <= root->nodeRangeZ_[0] && boxpoint.maxVertex[2] > root->nodeRangeZ_[1]) {
      flatten(root, storage, NOT_RECORD);
      return;
    }
    if (boxpoint.minVertex[0] <= root->point_.x() && boxpoint.maxVertex[0] > root->point_.x() &&
        boxpoint.minVertex[1] <= root->point_.y() && boxpoint.maxVertex[1] > root->point_.y() &&
        boxpoint.minVertex[2] <= root->point_.z() && boxpoint.maxVertex[2] > root->point_.z()) {
      if (!root->pointDeleted_) storage.push_back(root->point_);
    }
    if ((rebuildNode_ == nullptr) || root->leftSon_ != *rebuildNode_) {
      searchRange(root->leftSon_, boxpoint, storage);
    } else {
      pthread_mutex_lock(&searchLock_);
      searchRange(root->leftSon_, boxpoint, storage);
      pthread_mutex_unlock(&searchLock_);
    }
    if ((rebuildNode_ == nullptr) || root->rightSon_ != *rebuildNode_) {
      searchRange(root->rightSon_, boxpoint, storage);
    } else {
      pthread_mutex_lock(&searchLock_);
      searchRange(root->rightSon_, boxpoint, storage);
      pthread_mutex_unlock(&searchLock_);
    }
  };
  void radiusSearch(KdTreeNode* root, PointType point_, double radius, PointVector& storage) {
    if (root == nullptr) return;
    pushDown(root);
    PointType rangeCenter;
    rangeCenter.x() = (root->nodeRangeX_[0] + root->nodeRangeX_[1]) * 0.5;
    rangeCenter.y() = (root->nodeRangeY_[0] + root->nodeRangeY_[1]) * 0.5;
    rangeCenter.z() = (root->nodeRangeZ_[0] + root->nodeRangeZ_[1]) * 0.5;
    double dist = sqrt(calculateDistance(rangeCenter, point_));
    if (dist > radius + sqrt(root->radiusSq_)) return;
    if (dist <= radius - sqrt(root->radiusSq_)) {
      flatten(root, storage, NOT_RECORD);
      return;
    }
    if (!root->pointDeleted_ && calculateDistance(root->point_, point_) <= radius * radius) {
      storage.push_back(root->point_);
    }
    if ((rebuildNode_ == nullptr) || root->leftSon_ != *rebuildNode_) {
      radiusSearch(root->leftSon_, point_, radius, storage);
    } else {
      pthread_mutex_lock(&searchLock_);
      radiusSearch(root->leftSon_, point_, radius, storage);
      pthread_mutex_unlock(&searchLock_);
    }
    if ((rebuildNode_ == nullptr) || root->rightSon_ != *rebuildNode_) {
      radiusSearch(root->rightSon_, point_, radius, storage);
    } else {
      pthread_mutex_lock(&searchLock_);
      radiusSearch(root->rightSon_, point_, radius, storage);
      pthread_mutex_unlock(&searchLock_);
    }
  };
  bool criterionCheck(KdTreeNode* root) {
    if (root->treeSize_ <= Minimal_Unbalanced_Tree_Size) {
      return false;
    }
    double evalBalance = 0.0f;
    double evalDelete = 0.0f;
    KdTreeNode* son = root->leftSon_;
    if (son == nullptr) son = root->rightSon_;
    evalDelete = double(root->invalidPointNum_) / root->treeSize_;
    evalBalance = double(son->treeSize_) / (root->treeSize_ - 1);
    if (evalDelete > deleteCriterion_) {
      return true;
    }
    if (evalBalance > balanceCriterion_ || evalBalance < 1 - balanceCriterion_) {
      return true;
    }
    return false;
  };

  void pushDown(KdTreeNode* root) {
    if (root == nullptr) return;
    OperationLoggerType operation;
    operation.op = PUSH_DOWN;
    operation.treeDeleted_ = root->treeDeleted_;
    operation.treeDeletedDownsample_ = root->treeDeletedDownsample_;
    if (root->needsPushDownLeft_ && root->leftSon_ != nullptr) {
      if (rebuildNode_ == nullptr || *rebuildNode_ != root->leftSon_) {
        root->leftSon_->treeDeletedDownsample_ |= root->treeDeletedDownsample_;
        root->leftSon_->pointDeletedDownsample_ |= root->treeDeletedDownsample_;
        root->leftSon_->treeDeleted_ = root->treeDeleted_ || root->leftSon_->treeDeletedDownsample_;
        root->leftSon_->pointDeleted_ = root->leftSon_->treeDeleted_ || root->leftSon_->pointDeletedDownsample_;
        if (root->treeDeletedDownsample_) root->leftSon_->downsampledDeleteNum_ = root->leftSon_->treeSize_;
        if (root->treeDeleted_)
          root->leftSon_->invalidPointNum_ = root->leftSon_->treeSize_;
        else
          root->leftSon_->invalidPointNum_ = root->leftSon_->downsampledDeleteNum_;
        root->leftSon_->needsPushDownLeft_ = true;
        root->leftSon_->needsPushDownRight_ = true;
        root->needsPushDownLeft_ = false;
      } else {
        pthread_mutex_lock(&isRebuildLock_);
        root->leftSon_->treeDeletedDownsample_ |= root->treeDeletedDownsample_;
        root->leftSon_->pointDeletedDownsample_ |= root->treeDeletedDownsample_;
        root->leftSon_->treeDeleted_ = root->treeDeleted_ || root->leftSon_->treeDeletedDownsample_;
        root->leftSon_->pointDeleted_ = root->leftSon_->treeDeleted_ || root->leftSon_->pointDeletedDownsample_;
        if (root->treeDeletedDownsample_) root->leftSon_->downsampledDeleteNum_ = root->leftSon_->treeSize_;
        if (root->treeDeleted_)
          root->leftSon_->invalidPointNum_ = root->leftSon_->treeSize_;
        else
          root->leftSon_->invalidPointNum_ = root->leftSon_->downsampledDeleteNum_;
        root->leftSon_->needsPushDownLeft_ = true;
        root->leftSon_->needsPushDownRight_ = true;
        if (isRebuild_) {
          pthread_mutex_lock(&rebuildLoggerLock_);
          rebuildLogger_.push(operation);
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        root->needsPushDownLeft_ = false;
        pthread_mutex_unlock(&isRebuildLock_);
      }
    }
    if (root->needsPushDownRight_ && root->rightSon_ != nullptr) {
      if (rebuildNode_ == nullptr || *rebuildNode_ != root->rightSon_) {
        root->rightSon_->treeDeletedDownsample_ |= root->treeDeletedDownsample_;
        root->rightSon_->pointDeletedDownsample_ |= root->treeDeletedDownsample_;
        root->rightSon_->treeDeleted_ = root->treeDeleted_ || root->rightSon_->treeDeletedDownsample_;
        root->rightSon_->pointDeleted_ = root->rightSon_->treeDeleted_ || root->rightSon_->pointDeletedDownsample_;
        if (root->treeDeletedDownsample_) root->rightSon_->downsampledDeleteNum_ = root->rightSon_->treeSize_;
        if (root->treeDeleted_)
          root->rightSon_->invalidPointNum_ = root->rightSon_->treeSize_;
        else
          root->rightSon_->invalidPointNum_ = root->rightSon_->downsampledDeleteNum_;
        root->rightSon_->needsPushDownLeft_ = true;
        root->rightSon_->needsPushDownRight_ = true;
        root->needsPushDownRight_ = false;
      } else {
        pthread_mutex_lock(&isRebuildLock_);
        root->rightSon_->treeDeletedDownsample_ |= root->treeDeletedDownsample_;
        root->rightSon_->pointDeletedDownsample_ |= root->treeDeletedDownsample_;
        root->rightSon_->treeDeleted_ = root->treeDeleted_ || root->rightSon_->treeDeletedDownsample_;
        root->rightSon_->pointDeleted_ = root->rightSon_->treeDeleted_ || root->rightSon_->pointDeletedDownsample_;
        if (root->treeDeletedDownsample_) root->rightSon_->downsampledDeleteNum_ = root->rightSon_->treeSize_;
        if (root->treeDeleted_)
          root->rightSon_->invalidPointNum_ = root->rightSon_->treeSize_;
        else
          root->rightSon_->invalidPointNum_ = root->rightSon_->downsampledDeleteNum_;
        root->rightSon_->needsPushDownLeft_ = true;
        root->rightSon_->needsPushDownRight_ = true;
        if (isRebuild_) {
          pthread_mutex_lock(&rebuildLoggerLock_);
          rebuildLogger_.push(operation);
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        root->needsPushDownRight_ = false;
        pthread_mutex_unlock(&isRebuildLock_);
      }
    }
  };

  void update(KdTreeNode* root) {
    KdTreeNode* leftSon = root->leftSon_;
    KdTreeNode* rightSon = root->rightSon_;
    double rangeTempX[2] = {INFINITY, -INFINITY};
    double rangeTempY[2] = {INFINITY, -INFINITY};
    double rangeTempZ[2] = {INFINITY, -INFINITY};
    // update Tree Size
    if (leftSon != nullptr && rightSon != nullptr) {
      root->treeSize_ = leftSon->treeSize_ + rightSon->treeSize_ + 1;
      root->invalidPointNum_ = leftSon->invalidPointNum_ + rightSon->invalidPointNum_ + (root->pointDeleted_ ? 1 : 0);
      root->downsampledDeleteNum_ =
          leftSon->downsampledDeleteNum_ + rightSon->downsampledDeleteNum_ + (root->pointDeletedDownsample_ ? 1 : 0);
      root->treeDeletedDownsample_ = leftSon->treeDeletedDownsample_ & rightSon->treeDeletedDownsample_ & root->pointDeletedDownsample_;
      root->treeDeleted_ = leftSon->treeDeleted_ && rightSon->treeDeleted_ && root->pointDeleted_;
      if (root->treeDeleted_ || (!leftSon->treeDeleted_ && !rightSon->treeDeleted_ && !root->pointDeleted_)) {
        rangeTempX[0] = min(min(leftSon->nodeRangeX_[0], rightSon->nodeRangeX_[0]), root->point_.x());
        rangeTempX[1] = max(max(leftSon->nodeRangeX_[1], rightSon->nodeRangeX_[1]), root->point_.x());
        rangeTempY[0] = min(min(leftSon->nodeRangeY_[0], rightSon->nodeRangeY_[0]), root->point_.y());
        rangeTempY[1] = max(max(leftSon->nodeRangeY_[1], rightSon->nodeRangeY_[1]), root->point_.y());
        rangeTempZ[0] = min(min(leftSon->nodeRangeZ_[0], rightSon->nodeRangeZ_[0]), root->point_.z());
        rangeTempZ[1] = max(max(leftSon->nodeRangeZ_[1], rightSon->nodeRangeZ_[1]), root->point_.z());
      } else {
        if (!leftSon->treeDeleted_) {
          rangeTempX[0] = min(rangeTempX[0], leftSon->nodeRangeX_[0]);
          rangeTempX[1] = max(rangeTempX[1], leftSon->nodeRangeX_[1]);
          rangeTempY[0] = min(rangeTempY[0], leftSon->nodeRangeY_[0]);
          rangeTempY[1] = max(rangeTempY[1], leftSon->nodeRangeY_[1]);
          rangeTempZ[0] = min(rangeTempZ[0], leftSon->nodeRangeZ_[0]);
          rangeTempZ[1] = max(rangeTempZ[1], leftSon->nodeRangeZ_[1]);
        }
        if (!rightSon->treeDeleted_) {
          rangeTempX[0] = min(rangeTempX[0], rightSon->nodeRangeX_[0]);
          rangeTempX[1] = max(rangeTempX[1], rightSon->nodeRangeX_[1]);
          rangeTempY[0] = min(rangeTempY[0], rightSon->nodeRangeY_[0]);
          rangeTempY[1] = max(rangeTempY[1], rightSon->nodeRangeY_[1]);
          rangeTempZ[0] = min(rangeTempZ[0], rightSon->nodeRangeZ_[0]);
          rangeTempZ[1] = max(rangeTempZ[1], rightSon->nodeRangeZ_[1]);
        }
        if (!root->pointDeleted_) {
          rangeTempX[0] = min(rangeTempX[0], root->point_.x());
          rangeTempX[1] = max(rangeTempX[1], root->point_.x());
          rangeTempY[0] = min(rangeTempY[0], root->point_.y());
          rangeTempY[1] = max(rangeTempY[1], root->point_.y());
          rangeTempZ[0] = min(rangeTempZ[0], root->point_.z());
          rangeTempZ[1] = max(rangeTempZ[1], root->point_.z());
        }
      }
    } else if (leftSon != nullptr) {
      root->treeSize_ = leftSon->treeSize_ + 1;
      root->invalidPointNum_ = leftSon->invalidPointNum_ + (root->pointDeleted_ ? 1 : 0);
      root->downsampledDeleteNum_ = leftSon->downsampledDeleteNum_ + (root->pointDeletedDownsample_ ? 1 : 0);
      root->treeDeletedDownsample_ = leftSon->treeDeletedDownsample_ & root->pointDeletedDownsample_;
      root->treeDeleted_ = leftSon->treeDeleted_ && root->pointDeleted_;
      if (root->treeDeleted_ || (!leftSon->treeDeleted_ && !root->pointDeleted_)) {
        rangeTempX[0] = min(leftSon->nodeRangeX_[0], root->point_.x());
        rangeTempX[1] = max(leftSon->nodeRangeX_[1], root->point_.x());
        rangeTempY[0] = min(leftSon->nodeRangeY_[0], root->point_.y());
        rangeTempY[1] = max(leftSon->nodeRangeY_[1], root->point_.y());
        rangeTempZ[0] = min(leftSon->nodeRangeZ_[0], root->point_.z());
        rangeTempZ[1] = max(leftSon->nodeRangeZ_[1], root->point_.z());
      } else {
        if (!leftSon->treeDeleted_) {
          rangeTempX[0] = min(rangeTempX[0], leftSon->nodeRangeX_[0]);
          rangeTempX[1] = max(rangeTempX[1], leftSon->nodeRangeX_[1]);
          rangeTempY[0] = min(rangeTempY[0], leftSon->nodeRangeY_[0]);
          rangeTempY[1] = max(rangeTempY[1], leftSon->nodeRangeY_[1]);
          rangeTempZ[0] = min(rangeTempZ[0], leftSon->nodeRangeZ_[0]);
          rangeTempZ[1] = max(rangeTempZ[1], leftSon->nodeRangeZ_[1]);
        }
        if (!root->pointDeleted_) {
          rangeTempX[0] = min(rangeTempX[0], root->point_.x());
          rangeTempX[1] = max(rangeTempX[1], root->point_.x());
          rangeTempY[0] = min(rangeTempY[0], root->point_.y());
          rangeTempY[1] = max(rangeTempY[1], root->point_.y());
          rangeTempZ[0] = min(rangeTempZ[0], root->point_.z());
          rangeTempZ[1] = max(rangeTempZ[1], root->point_.z());
        }
      }

    } else if (rightSon != nullptr) {
      root->treeSize_ = rightSon->treeSize_ + 1;
      root->invalidPointNum_ = rightSon->invalidPointNum_ + (root->pointDeleted_ ? 1 : 0);
      root->downsampledDeleteNum_ = rightSon->downsampledDeleteNum_ + (root->pointDeletedDownsample_ ? 1 : 0);
      root->treeDeletedDownsample_ = rightSon->treeDeletedDownsample_ & root->pointDeletedDownsample_;
      root->treeDeleted_ = rightSon->treeDeleted_ && root->pointDeleted_;
      if (root->treeDeleted_ || (!rightSon->treeDeleted_ && !root->pointDeleted_)) {
        rangeTempX[0] = min(rightSon->nodeRangeX_[0], root->point_.x());
        rangeTempX[1] = max(rightSon->nodeRangeX_[1], root->point_.x());
        rangeTempY[0] = min(rightSon->nodeRangeY_[0], root->point_.y());
        rangeTempY[1] = max(rightSon->nodeRangeY_[1], root->point_.y());
        rangeTempZ[0] = min(rightSon->nodeRangeZ_[0], root->point_.z());
        rangeTempZ[1] = max(rightSon->nodeRangeZ_[1], root->point_.z());
      } else {
        if (!rightSon->treeDeleted_) {
          rangeTempX[0] = min(rangeTempX[0], rightSon->nodeRangeX_[0]);
          rangeTempX[1] = max(rangeTempX[1], rightSon->nodeRangeX_[1]);
          rangeTempY[0] = min(rangeTempY[0], rightSon->nodeRangeY_[0]);
          rangeTempY[1] = max(rangeTempY[1], rightSon->nodeRangeY_[1]);
          rangeTempZ[0] = min(rangeTempZ[0], rightSon->nodeRangeZ_[0]);
          rangeTempZ[1] = max(rangeTempZ[1], rightSon->nodeRangeZ_[1]);
        }
        if (!root->pointDeleted_) {
          rangeTempX[0] = min(rangeTempX[0], root->point_.x());
          rangeTempX[1] = max(rangeTempX[1], root->point_.x());
          rangeTempY[0] = min(rangeTempY[0], root->point_.y());
          rangeTempY[1] = max(rangeTempY[1], root->point_.y());
          rangeTempZ[0] = min(rangeTempZ[0], root->point_.z());
          rangeTempZ[1] = max(rangeTempZ[1], root->point_.z());
        }
      }
    } else {
      root->treeSize_ = 1;
      root->invalidPointNum_ = (root->pointDeleted_ ? 1 : 0);
      root->downsampledDeleteNum_ = (root->pointDeletedDownsample_ ? 1 : 0);
      root->treeDeletedDownsample_ = root->pointDeletedDownsample_;
      root->treeDeleted_ = root->pointDeleted_;
      rangeTempX[0] = root->point_.x();
      rangeTempX[1] = root->point_.x();
      rangeTempY[0] = root->point_.y();
      rangeTempY[1] = root->point_.y();
      rangeTempZ[0] = root->point_.z();
      rangeTempZ[1] = root->point_.z();
    }
    memcpy(root->nodeRangeX_, rangeTempX, sizeof(rangeTempX));
    memcpy(root->nodeRangeY_, rangeTempY, sizeof(rangeTempY));
    memcpy(root->nodeRangeZ_, rangeTempZ, sizeof(rangeTempZ));
    double x_L = (root->nodeRangeX_[1] - root->nodeRangeX_[0]) * 0.5;
    double y_L = (root->nodeRangeY_[1] - root->nodeRangeY_[0]) * 0.5;
    double z_L = (root->nodeRangeZ_[1] - root->nodeRangeZ_[0]) * 0.5;
    root->radiusSq_ = x_L * x_L + y_L * y_L + z_L * z_L;
    if (leftSon != nullptr) leftSon->father_ = root;
    if (rightSon != nullptr) rightSon->father_ = root;
    if (root == rootNode_ && root->treeSize_ > 3) {
      KdTreeNode* son = root->leftSon_;
      if (son == nullptr) son = root->rightSon_;
      double tmp_bal = double(son->treeSize_) / (root->treeSize_ - 1);
      root->alphaDel_ = double(root->invalidPointNum_) / root->treeSize_;
      root->alphaBal_ = (tmp_bal >= 0.5 - EPSS) ? tmp_bal : 1 - tmp_bal;
    }
  };

  void deleteTreeNodes(KdTreeNode** root) {
    if (*root == nullptr) return;
    pushDown(*root);
    deleteTreeNodes(&(*root)->leftSon_);
    deleteTreeNodes(&(*root)->rightSon_);

    pthread_mutex_destroy(&(*root)->pushDownLock_);
    delete *root;
    *root = nullptr;
  };

  bool isSamePoint(PointType a, PointType b) {
    return (fabs(a.x() - b.x()) < EPSS && fabs(a.y() - b.y()) < EPSS && fabs(a.z() - b.z()) < EPSS);
  };

  double calculateDistance(PointType a, PointType b) {
    double dist = 0.0f;
    dist = (a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y()) + (a.z() - b.z()) * (a.z() - b.z());
    return dist;
  };

  double calculateBoxDistance(KdTreeNode* node, PointType point_) {
    if (node == nullptr) return INFINITY;
    double minDist = 0.0;
    if (point_.x() < node->nodeRangeX_[0]) minDist += (point_.x() - node->nodeRangeX_[0]) * (point_.x() - node->nodeRangeX_[0]);
    if (point_.x() > node->nodeRangeX_[1]) minDist += (point_.x() - node->nodeRangeX_[1]) * (point_.x() - node->nodeRangeX_[1]);
    if (point_.y() < node->nodeRangeY_[0]) minDist += (point_.y() - node->nodeRangeY_[0]) * (point_.y() - node->nodeRangeY_[0]);
    if (point_.y() > node->nodeRangeY_[1]) minDist += (point_.y() - node->nodeRangeY_[1]) * (point_.y() - node->nodeRangeY_[1]);
    if (point_.z() < node->nodeRangeZ_[0]) minDist += (point_.z() - node->nodeRangeZ_[0]) * (point_.z() - node->nodeRangeZ_[0]);
    if (point_.z() > node->nodeRangeZ_[1]) minDist += (point_.z() - node->nodeRangeZ_[1]) * (point_.z() - node->nodeRangeZ_[1]);
    return minDist;
  };

  static bool compareX(PointType a, PointType b) { return a.x() < b.x(); };
  static bool compareY(PointType a, PointType b) { return a.y() < b.y(); };
  static bool compareZ(PointType a, PointType b) { return a.z() < b.z(); };

 public:
  explicit KdTree(double delete_param = 0.5, double balance_param = 0.6, double box_length = 0.2) {
    deleteCriterion_ = delete_param;
    balanceCriterion_ = balance_param;
    downsampleSize_ = box_length;
    rebuildLogger_.clear();
    isDone_ = false;
    startThread();
  };

  ~KdTree() {
    stopThread();
    deleteTreeNodes(&rootNode_);
    PointVector().swap(pointStorage_);
    rebuildLogger_.clear();
  };

  void setDeleteCriterion(double delete_param) { deleteCriterion_ = delete_param; };
  void setBalanceCriterion(double balance_param) { balanceCriterion_ = balance_param; };
  void setDownsampleSize(double box_length) { downsampleSize_ = box_length; };
  void initializeKdTree(double delete_param = 0.5, double balance_param = 0.7, double box_length = 0.2) {
    setDeleteCriterion(delete_param);
    setBalanceCriterion(balance_param);
    setDownsampleSize(box_length);
  };

  int size() {
    int s = 0;
    if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
      if (rootNode_ != nullptr) {
        return rootNode_->treeSize_;
      } else {
        return 0;
      }
    } else {
      if (!pthread_mutex_trylock(&isRebuildLock_)) {
        s = rootNode_->treeSize_;
        pthread_mutex_unlock(&isRebuildLock_);
        return s;
      } else {
        return tmpTreeSize_;
      }
    }
  };

  int numValidNodes() {
    int s = 0;
    if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
      if (rootNode_ != nullptr)
        return (rootNode_->treeSize_ - rootNode_->invalidPointNum_);
      else
        return 0;
    } else {
      if (!pthread_mutex_trylock(&isRebuildLock_)) {
        s = rootNode_->treeSize_ - rootNode_->invalidPointNum_;
        pthread_mutex_unlock(&isRebuildLock_);
        return s;
      } else {
        return -1;
      }
    }
  };

  void rootAlpha(double& alphaBal_, double& alphaDel_) {
    if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
      alphaBal_ = rootNode_->alphaBal_;
      alphaDel_ = rootNode_->alphaDel_;
      return;
    } else {
      if (!pthread_mutex_trylock(&isRebuildLock_)) {
        alphaBal_ = rootNode_->alphaBal_;
        alphaDel_ = rootNode_->alphaDel_;
        pthread_mutex_unlock(&isRebuildLock_);
        return;
      } else {
        alphaBal_ = tmpAlphaBal_;
        alphaDel_ = tmpAlphaDel_;
        return;
      }
    }
  };

  void build(PointVector point_cloud) {
    if (rootNode_ != nullptr) {
      deleteTreeNodes(&rootNode_);
    }
    if (point_cloud.size() == 0) return;
    STATIC_ROOT_NODE = new KdTreeNode;
    initializeTreeNode(STATIC_ROOT_NODE);
    buildTree(&STATIC_ROOT_NODE->leftSon_, 0, point_cloud.size() - 1, point_cloud);
    update(STATIC_ROOT_NODE);
    STATIC_ROOT_NODE->treeSize_ = 0;
    rootNode_ = STATIC_ROOT_NODE->leftSon_;
  };

  void searchNearest(PointType point_, int k_nearest, PointVector& Nearest_Points, vector<double>& Point_Distance,
                     double maxDistance = INFINITY) {
    ManualHeap q(2 * k_nearest);
    q.clear();
    vector<double>().swap(Point_Distance);
    if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
      search(rootNode_, k_nearest, point_, q, maxDistance);
    } else {
      pthread_mutex_lock(&searchLock_);
      while (searchLockCounter_ == -1) {
        pthread_mutex_unlock(&searchLock_);
        usleep(1);
        pthread_mutex_lock(&searchLock_);
      }
      searchLockCounter_ += 1;
      pthread_mutex_unlock(&searchLock_);
      search(rootNode_, k_nearest, point_, q, maxDistance);
      pthread_mutex_lock(&searchLock_);
      searchLockCounter_ -= 1;
      pthread_mutex_unlock(&searchLock_);
    }
    int k_found = min(k_nearest, int(q.size()));
    PointVector().swap(Nearest_Points);
    vector<double>().swap(Point_Distance);
    for (int i = 0; i < k_found; i++) {
      Nearest_Points.insert(Nearest_Points.begin(), q.top().point_);
      Point_Distance.insert(Point_Distance.begin(), q.top().dist_);
      q.pop();
    }
  };

  void searchBox(const BoxPointType& Box_of_Point, PointVector& storage) {
    storage.clear();
    searchRange(rootNode_, Box_of_Point, storage);
  };

  void searchRadius(PointType point_, const double radius, PointVector& storage) {
    storage.clear();
    radiusSearch(rootNode_, point_, radius, storage);
  };

  int addPoints(const PointVector& PointToAdd, bool shouldDownsample) {
    int NewPointSize = PointToAdd.size();
    int tree_size = size();
    BoxPointType Box_of_Point{};
    PointType downsample_result, mid_point;
    bool downsample_switch = shouldDownsample && DOWNSAMPLE_SWITCH;
    double minDist, tmp_dist;
    int counter = 0;
    for (int i = 0; i < PointToAdd.size(); i++) {
      if (downsample_switch) {
        Box_of_Point.minVertex[0] = floor(PointToAdd[i].x() / downsampleSize_) * downsampleSize_;
        Box_of_Point.maxVertex[0] = Box_of_Point.minVertex[0] + downsampleSize_;
        Box_of_Point.minVertex[1] = floor(PointToAdd[i].y() / downsampleSize_) * downsampleSize_;
        Box_of_Point.maxVertex[1] = Box_of_Point.minVertex[1] + downsampleSize_;
        Box_of_Point.minVertex[2] = floor(PointToAdd[i].z() / downsampleSize_) * downsampleSize_;
        Box_of_Point.maxVertex[2] = Box_of_Point.minVertex[2] + downsampleSize_;
        mid_point.x() = Box_of_Point.minVertex[0] + (Box_of_Point.maxVertex[0] - Box_of_Point.minVertex[0]) / 2.0;
        mid_point.y() = Box_of_Point.minVertex[1] + (Box_of_Point.maxVertex[1] - Box_of_Point.minVertex[1]) / 2.0;
        mid_point.z() = Box_of_Point.minVertex[2] + (Box_of_Point.maxVertex[2] - Box_of_Point.minVertex[2]) / 2.0;
        PointVector().swap(downsampledPoints_);
        searchRange(rootNode_, Box_of_Point, downsampledPoints_);
        minDist = calculateDistance(PointToAdd[i], mid_point);
        downsample_result = PointToAdd[i];
        for (int index = 0; index < downsampledPoints_.size(); index++) {
          tmp_dist = calculateDistance(downsampledPoints_[index], mid_point);
          if (tmp_dist < minDist) {
            minDist = tmp_dist;
            downsample_result = downsampledPoints_[index];
          }
        }
        if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
          if (downsampledPoints_.size() > 1 || isSamePoint(PointToAdd[i], downsample_result)) {
            if (downsampledPoints_.size() > 0) deleteRange(&rootNode_, Box_of_Point, true, true);
            addPoint(&rootNode_, downsample_result, true, rootNode_->divisionAxis_);
            counter++;
          }
        } else {
          if (downsampledPoints_.size() > 1 || isSamePoint(PointToAdd[i], downsample_result)) {
            OperationLoggerType operation_delete, operation;
            operation_delete.boxpoint = Box_of_Point;
            operation_delete.op = DOWNSAMPLE_DELETE;
            operation.point_ = downsample_result;
            operation.op = ADD_POINT;
            pthread_mutex_lock(&isRebuildLock_);
            if (downsampledPoints_.size() > 0) deleteRange(&rootNode_, Box_of_Point, false, true);
            addPoint(&rootNode_, downsample_result, false, rootNode_->divisionAxis_);
            counter++;
            if (isRebuild_) {
              pthread_mutex_lock(&rebuildLoggerLock_);
              if (downsampledPoints_.size() > 0) rebuildLogger_.push(operation_delete);
              rebuildLogger_.push(operation);
              pthread_mutex_unlock(&rebuildLoggerLock_);
            }
            pthread_mutex_unlock(&isRebuildLock_);
          }
        }
      } else {
        if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
          addPoint(&rootNode_, PointToAdd[i], true, rootNode_->divisionAxis_);
        } else {
          OperationLoggerType operation;
          operation.point_ = PointToAdd[i];
          operation.op = ADD_POINT;
          pthread_mutex_lock(&isRebuildLock_);
          addPoint(&rootNode_, PointToAdd[i], false, rootNode_->divisionAxis_);
          if (isRebuild_) {
            pthread_mutex_lock(&rebuildLoggerLock_);
            rebuildLogger_.push(operation);
            pthread_mutex_unlock(&rebuildLoggerLock_);
          }
          pthread_mutex_unlock(&isRebuildLock_);
        }
      }
    }
    return counter;
  };

  int addPoint(const PointType& pointToAdd, bool shouldDownsample) {
    PointVector pt = {pointToAdd};
    return addPoints(pt, shouldDownsample);
  };

  void addPointBoxes(vector<BoxPointType>& boxPoints) {
    for (const auto boxPt : boxPoints) {
      if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
        addRange(&rootNode_, boxPt, true);
      } else {
        OperationLoggerType operation;
        operation.boxpoint = boxPt;
        operation.op = ADD_BOX;
        pthread_mutex_lock(&isRebuildLock_);
        addRange(&rootNode_, boxPt, false);
        if (isRebuild_) {
          pthread_mutex_lock(&rebuildLoggerLock_);
          rebuildLogger_.push(operation);
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        pthread_mutex_unlock(&isRebuildLock_);
      }
    }
  };
  void deletePoints(PointVector& PointToDel) {
    for (int i = 0; i < PointToDel.size(); i++) {
      if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
        deletePoint(&rootNode_, PointToDel[i], true);
      } else {
        OperationLoggerType operation;
        operation.point_ = PointToDel[i];
        operation.op = DELETE_POINT;
        pthread_mutex_lock(&isRebuildLock_);
        deletePoint(&rootNode_, PointToDel[i], false);
        if (isRebuild_) {
          pthread_mutex_lock(&rebuildLoggerLock_);
          rebuildLogger_.push(operation);
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        pthread_mutex_unlock(&isRebuildLock_);
      }
    }
  };
  int deletePointBoxes(vector<BoxPointType>& boxPoints) {
    int counter = 0;
    for (const auto& boxPt : boxPoints) {
      if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
        counter += deleteRange(&rootNode_, boxPt, true, false);
      } else {
        OperationLoggerType operation;
        operation.boxpoint = boxPt;
        operation.op = DELETE_BOX;
        pthread_mutex_lock(&isRebuildLock_);
        counter += deleteRange(&rootNode_, boxPt, false, false);
        if (isRebuild_) {
          pthread_mutex_lock(&rebuildLoggerLock_);
          rebuildLogger_.push(operation);
          pthread_mutex_unlock(&rebuildLoggerLock_);
        }
        pthread_mutex_unlock(&isRebuildLock_);
      }
    }
    return counter;
  };
  void flatten(KdTreeNode* root, PointVector& storage, DeletedPointStorage storage_type) {
    if (root == nullptr) return;
    pushDown(root);
    if (!root->pointDeleted_) {
      storage.push_back(root->point_);
    }
    flatten(root->leftSon_, storage, storage_type);
    flatten(root->rightSon_, storage, storage_type);
    switch (storage_type) {
      case NOT_RECORD:
        break;
      case DELETE_POINTS_REC:
        if (root->pointDeleted_ && !root->pointDeletedDownsample_) {
          deletedPoints_.push_back(root->point_);
        }
        break;
      case MULTI_THREAD_REC:
        if (root->pointDeleted_ && !root->pointDeletedDownsample_) {
          deletedPointsMultithread_.push_back(root->point_);
        }
        break;
      default:
        break;
    }
  };
  void getRemovedPoints(PointVector& removed_points) {
    pthread_mutex_lock(&deletePointsCacheLock_);
    for (int i = 0; i < deletedPoints_.size(); i++) {
      removed_points.push_back(deletedPoints_[i]);
    }
    for (int i = 0; i < deletedPointsMultithread_.size(); i++) {
      removed_points.push_back(deletedPointsMultithread_[i]);
    }
    deletedPoints_.clear();
    deletedPointsMultithread_.clear();
    pthread_mutex_unlock(&deletePointsCacheLock_);
  };
  BoxPointType treeRange() {
    BoxPointType range{};
    if (rebuildNode_ == nullptr || *rebuildNode_ != rootNode_) {
      if (rootNode_ != nullptr) {
        range.minVertex[0] = rootNode_->nodeRangeX_[0];
        range.minVertex[1] = rootNode_->nodeRangeY_[0];
        range.minVertex[2] = rootNode_->nodeRangeZ_[0];
        range.maxVertex[0] = rootNode_->nodeRangeX_[1];
        range.maxVertex[1] = rootNode_->nodeRangeY_[1];
        range.maxVertex[2] = rootNode_->nodeRangeZ_[1];
      } else {
        memset(&range, 0, sizeof(range));
      }
    } else {
      if (!pthread_mutex_trylock(&isRebuildLock_)) {
        range.minVertex[0] = rootNode_->nodeRangeX_[0];
        range.minVertex[1] = rootNode_->nodeRangeY_[0];
        range.minVertex[2] = rootNode_->nodeRangeZ_[0];
        range.maxVertex[0] = rootNode_->nodeRangeX_[1];
        range.maxVertex[1] = rootNode_->nodeRangeY_[1];
        range.maxVertex[2] = rootNode_->nodeRangeZ_[1];
        pthread_mutex_unlock(&isRebuildLock_);
      } else {
        memset(&range, 0, sizeof(range));
      }
    }
    return range;
  };

  PointVector pointStorage_;
  KdTreeNode* rootNode_ = nullptr;
  int maxQueueSize_ = 0;
};

}  // namespace ikdTree